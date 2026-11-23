//! Face-Aware Liquify: the Eyes, Nose, Mouth and Face Shape sliders of Filter › Liquify.
//!
//! Each detected face ([`crate::faces`], five landmarks plus a box) carries its slider values,
//! `-100..=100`. Every slider is a smooth local warp anchored to the landmarks: a scale, rotation
//! or shift of a feature, faded to nothing over a soft ellipse around it so nothing tears. The
//! warps are added into the [`LiquifyField`]'s inverse displacement after the brush strokes, so
//! the mesh, the freeze mask, the preview and the smart filter treat them like any other edit.
//!
//! The landmark model has five points (eyes, nose tip, mouth corners), not a face outline, so the
//! forehead, chin and jaw are placed from the face's proportions: those three are estimates.
//!
//! Coordinates are document pixels. The face frame has `u` along the eye line (image left to
//! right) and `v` perpendicular to it, pointing down the face, so a tilted head works the same.

use serde::{Deserialize, Serialize};

use crate::liquify::LiquifyField;

/// A face's geometry, as detected, in document pixels.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FaceShape {
    /// Left, top, width, height.
    pub rect: [f64; 4],
    /// Eye on the image's left, eye on the image's right, nose tip, mouth corner on the image's
    /// left, mouth corner on the image's right (the detector's order: the face's right eye,
    /// left eye, nose, right and left mouth corners).
    pub landmarks: [[f64; 2]; 5],
}

/// The slider values for one face, each `-100..=100` (0 = unchanged). The eye sliders are
/// `[eye on the image's left, eye on the image's right]`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct FaceSliders {
    pub eye_size: [f64; 2],
    pub eye_height: [f64; 2],
    pub eye_width: [f64; 2],
    pub eye_tilt: [f64; 2],
    pub eye_distance: f64,
    pub nose_height: f64,
    pub nose_width: f64,
    pub smile: f64,
    pub upper_lip: f64,
    pub lower_lip: f64,
    pub mouth_width: f64,
    pub mouth_height: f64,
    pub forehead: f64,
    pub chin_height: f64,
    pub jawline: f64,
    pub face_width: f64,
}

/// One face and its sliders, as stored in the Liquify params (`faces`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FaceEdit {
    #[serde(flatten)]
    pub face: FaceShape,
    #[serde(flatten)]
    pub sliders: FaceSliders,
}

impl FaceSliders {
    fn values(&self) -> impl Iterator<Item = f64> {
        let pairs = [self.eye_size, self.eye_height, self.eye_width, self.eye_tilt];
        let singles = [
            self.eye_distance,
            self.nose_height,
            self.nose_width,
            self.smile,
            self.upper_lip,
            self.lower_lip,
            self.mouth_width,
            self.mouth_height,
            self.forehead,
            self.chin_height,
            self.jawline,
            self.face_width,
        ];
        pairs.into_iter().flatten().chain(singles)
    }

    /// Whether every slider is 0.
    pub fn is_neutral(&self) -> bool {
        self.values().all(|v| v == 0.0)
    }
}

impl FaceEdit {
    /// Checks the values: finite, sliders within `-100..=100`, a face of some size.
    pub fn validate(&self) -> Result<(), String> {
        let f = &self.face;
        if !f.rect.iter().chain(f.landmarks.iter().flatten()).all(|v| v.is_finite()) {
            return Err("face coordinates must be finite".into());
        }
        if !(f.rect[2] > 1.0 && f.rect[3] > 1.0) {
            return Err("the face box must be wider and taller than 1 px".into());
        }
        if dist(f.landmarks[0], f.landmarks[1]) < 1.0 {
            return Err("the eyes must be at least 1 px apart".into());
        }
        if !self.sliders.values().all(|v| v.is_finite() && (-100.0..=100.0).contains(&v)) {
            return Err("face sliders must be -100..100".into());
        }
        Ok(())
    }
}

fn dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

fn mid(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5]
}

/// The face's own axes: `u` along the eye line, `v` down the face.
#[derive(Clone, Copy)]
struct Frame {
    u: [f64; 2],
    v: [f64; 2],
}

impl Frame {
    fn to_local(self, d: [f64; 2]) -> [f64; 2] {
        [d[0] * self.u[0] + d[1] * self.u[1], d[0] * self.v[0] + d[1] * self.v[1]]
    }
    fn to_doc(self, l: [f64; 2]) -> [f64; 2] {
        [l[0] * self.u[0] + l[1] * self.v[0], l[0] * self.u[1] + l[1] * self.v[1]]
    }
    /// A point `du` along the eye line and `dv` down the face from `p`.
    fn at(self, p: [f64; 2], du: f64, dv: f64) -> [f64; 2] {
        let d = self.to_doc([du, dv]);
        [p[0] + d[0], p[1] + d[1]]
    }
}

/// Which side of a feature's centre line a warp reaches.
#[derive(Clone, Copy, PartialEq)]
enum Half {
    Both,
    /// Only above the centre (toward the forehead).
    Above,
    /// Only below the centre (toward the chin).
    Below,
}

/// One local warp, in the face frame around `centre`: a linear map `m` (applied to the local
/// offset) plus a shift `t`, at full strength in the middle and fading out at the ellipse
/// `radii` (local u, v). Output pixel `p` shows the source pixel the warp's inverse sends it to.
struct Warp {
    centre: [f64; 2],
    radii: [f64; 2],
    /// Forward linear map in local coordinates, row-major.
    m: [[f64; 2]; 2],
    /// Forward shift in local coordinates.
    t: [f64; 2],
    half: Half,
}

impl Warp {
    fn scale(centre: [f64; 2], radii: [f64; 2], su: f64, sv: f64) -> Self {
        Warp { centre, radii, m: [[su, 0.0], [0.0, sv]], t: [0.0; 2], half: Half::Both }
    }
    fn shift(centre: [f64; 2], radii: [f64; 2], tu: f64, tv: f64) -> Self {
        Warp { centre, radii, m: [[1.0, 0.0], [0.0, 1.0]], t: [tu, tv], half: Half::Both }
    }
    fn rotate(centre: [f64; 2], radii: [f64; 2], a: f64) -> Self {
        let (s, c) = a.sin_cos();
        Warp { centre, radii, m: [[c, -s], [s, c]], t: [0.0; 2], half: Half::Both }
    }
    fn half(mut self, h: Half) -> Self {
        self.half = h;
        self
    }

    /// Smooth weight, 1 at the centre and 0 (with zero slope) at the ellipse. A one-sided warp
    /// also fades in across its centre line (over a third of its height) instead of starting
    /// abruptly there: a cut would tear the image wherever the warp moves pixels along the line.
    fn weight(&self, l: [f64; 2]) -> f64 {
        let (a, b) = (l[0] / self.radii[0], l[1] / self.radii[1]);
        let t2 = a * a + b * b;
        if t2 >= 1.0 {
            return 0.0;
        }
        let k = 1.0 - t2;
        let side = match self.half {
            Half::Both => 1.0,
            Half::Above => smoothstep(-b * 3.0),
            Half::Below => smoothstep(b * 3.0),
        };
        k * k * side
    }

    /// The inverse displacement this warp gives output offset `l` (local), in local units.
    fn inverse(&self, l: [f64; 2]) -> [f64; 2] {
        let w = self.weight(l);
        if w == 0.0 {
            return [0.0; 2];
        }
        // Blend the map toward identity by the weight, then invert it: source = M_w⁻¹ (l - t_w).
        let m = [[1.0 + (self.m[0][0] - 1.0) * w, self.m[0][1] * w], [self.m[1][0] * w, 1.0 + (self.m[1][1] - 1.0) * w]];
        let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
        if det.abs() < 1e-6 {
            return [0.0; 2];
        }
        let q = [l[0] - self.t[0] * w, l[1] - self.t[1] * w];
        let src = [(m[1][1] * q[0] - m[0][1] * q[1]) / det, (-m[1][0] * q[0] + m[0][0] * q[1]) / det];
        [src[0] - l[0], src[1] - l[1]]
    }
}

/// 0 below 0, 1 above 1, smooth (zero slope at both ends) between.
fn smoothstep(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// The warps one face's sliders make, in face-frame terms.
fn warps(e: &FaceEdit) -> (Frame, Vec<Warp>) {
    let s = &e.sliders;
    let lm = e.face.landmarks;
    let (le, re, nose, ml, mr) = (lm[0], lm[1], lm[2], lm[3], lm[4]);
    let iod = dist(le, re);
    let ux = [(re[0] - le[0]) / iod, (re[1] - le[1]) / iod];
    let f = Frame { u: ux, v: [-ux[1], ux[0]] };
    let eyes = mid(le, re);
    let mouth = mid(ml, mr);
    let mouth_w = dist(ml, mr).max(iod * 0.3);
    // How far the mouth sits below the eyes, along the face.
    let drop = f.to_local([mouth[0] - eyes[0], mouth[1] - eyes[1]])[1].max(iod * 0.5);
    let face_w = e.face.rect[2].max(iod * 1.5);
    let chin = f.at(mouth, 0.0, drop * 0.55);
    let x = |v: f64| v / 100.0;
    let mut out = Vec::new();

    // Eyes: [image left, image right]; `side` is -1 for the left eye, +1 for the right, so
    // tilt and distance mirror across the face.
    let eye_r = [iod * 0.42, iod * 0.32];
    for (k, (c, side)) in [(le, -1.0), (re, 1.0)].into_iter().enumerate() {
        let size = 1.0 + 0.35 * x(s.eye_size[k]);
        if size != 1.0 {
            out.push(Warp::scale(c, eye_r, size, size));
        }
        if s.eye_height[k] != 0.0 {
            out.push(Warp::scale(c, eye_r, 1.0, 1.0 + 0.45 * x(s.eye_height[k])));
        }
        if s.eye_width[k] != 0.0 {
            out.push(Warp::scale(c, eye_r, 1.0 + 0.3 * x(s.eye_width[k]), 1.0));
        }
        if s.eye_tilt[k] != 0.0 {
            // Positive raises the outer corner (outer = away from the nose).
            out.push(Warp::rotate(c, eye_r, -side * 0.3 * x(s.eye_tilt[k])));
        }
        if s.eye_distance != 0.0 {
            out.push(Warp::shift(c, [iod * 0.5, iod * 0.38], side * 0.12 * iod * x(s.eye_distance), 0.0));
        }
    }

    // Nose: positive Nose Height shortens it (the tip moves up toward the eyes).
    let nose_r = [iod * 0.45, iod * 0.4];
    if s.nose_height != 0.0 {
        out.push(Warp::shift(nose, nose_r, 0.0, -0.12 * iod * x(s.nose_height)));
    }
    if s.nose_width != 0.0 {
        out.push(Warp::scale(nose, nose_r, 1.0 + 0.3 * x(s.nose_width), 1.0));
    }

    // Mouth.
    if s.smile != 0.0 {
        for c in [ml, mr] {
            out.push(Warp::shift(c, [mouth_w * 0.45, mouth_w * 0.4], 0.0, -0.08 * iod * x(s.smile)));
        }
    }
    let lip_r = [mouth_w * 0.85, iod * 0.45];
    if s.upper_lip != 0.0 {
        out.push(Warp::scale(mouth, lip_r, 1.0, 1.0 + 0.45 * x(s.upper_lip)).half(Half::Above));
    }
    if s.lower_lip != 0.0 {
        // Negative thickens the lower lip.
        out.push(Warp::scale(mouth, lip_r, 1.0, 1.0 - 0.45 * x(s.lower_lip)).half(Half::Below));
    }
    if s.mouth_width != 0.0 {
        out.push(Warp::scale(mouth, [mouth_w * 1.05, iod * 0.45], 1.0 + 0.25 * x(s.mouth_width), 1.0));
    }
    if s.mouth_height != 0.0 {
        out.push(Warp::scale(mouth, [mouth_w * 0.85, iod * 0.45], 1.0, 1.0 + 0.4 * x(s.mouth_height)));
    }

    // Face shape (estimated from the proportions; see the module docs).
    if s.forehead != 0.0 {
        let c = f.at(eyes, 0.0, -iod * 0.25);
        out.push(Warp::scale(c, [face_w * 0.7, drop * 2.2], 1.0, 1.0 + 0.18 * x(s.forehead)).half(Half::Above));
    }
    if s.chin_height != 0.0 {
        // Positive raises (shortens) the chin.
        out.push(Warp::shift(chin, [face_w * 0.45, drop * 0.75], 0.0, -0.12 * iod * x(s.chin_height)));
    }
    if s.jawline != 0.0 {
        let c = f.at(mouth, 0.0, drop * 0.2);
        out.push(Warp::scale(c, [face_w * 0.8, drop * 0.85], 1.0 + 0.14 * x(s.jawline), 1.0).half(Half::Below));
    }
    if s.face_width != 0.0 {
        out.push(Warp::scale(nose, [face_w * 0.8, drop * 1.6], 1.0 + 0.12 * x(s.face_width), 1.0));
    }
    (f, out)
}

impl LiquifyField {
    /// Adds a face's slider warps to the field (frozen areas stay put).
    pub fn apply_face(&mut self, e: &FaceEdit) {
        if e.sliders.is_neutral() || e.validate().is_err() {
            return;
        }
        let (frame, list) = warps(e);
        for w in &list {
            let reach = w.radii[0].max(w.radii[1]);
            let (i0, j0, i1, j1) = match self.node_span(w.centre, reach) {
                Some(s) => s,
                None => continue,
            };
            for j in j0..=j1 {
                for i in i0..=i1 {
                    let p = self.node_pos(i, j);
                    let l = frame.to_local([p[0] - w.centre[0], p[1] - w.centre[1]]);
                    let dl = w.inverse(l);
                    if dl == [0.0; 2] {
                        continue;
                    }
                    let k = j * self.w + i;
                    let free = f64::from(1.0 - self.freeze[k]);
                    let d = frame.to_doc(dl);
                    self.d[k][0] += (d[0] * free) as f32;
                    self.d[k][1] += (d[1] * free) as f32;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_geom::Rect;

    fn face() -> FaceShape {
        // An upright face: eyes at y 100, nose at 140, mouth at 175.
        FaceShape { rect: [50.0, 50.0, 200.0, 220.0], landmarks: [[110.0, 100.0], [190.0, 100.0], [150.0, 140.0], [175.0, 175.0], [125.0, 175.0]] }
    }

    fn field(s: FaceSliders) -> LiquifyField {
        let mut f = LiquifyField::new(Rect::new(0, 0, 300, 320), 2.0);
        f.apply_face(&FaceEdit { face: face(), sliders: s });
        f
    }

    #[test]
    fn neutral_sliders_change_nothing() {
        assert!(field(FaceSliders::default()).is_identity());
    }

    #[test]
    fn bigger_eyes_pull_from_closer_to_the_eye_centre() {
        let f = field(FaceSliders { eye_size: [100.0, 0.0], ..Default::default() });
        // A point right of the left eye shows a source nearer the eye (magnified).
        let d = f.sample(130.0, 100.0);
        assert!(d[0] < -1.0, "{d:?}");
        // The right eye was left alone.
        let r = f.sample(210.0, 100.0);
        assert!(r[0].abs() < 1e-3 && r[1].abs() < 1e-3, "{r:?}");
    }

    #[test]
    fn smile_lifts_the_mouth_corners() {
        let f = field(FaceSliders { smile: 100.0, ..Default::default() });
        // At a corner the output shows pixels from below: the corner moved up.
        assert!(f.sample(125.0, 175.0)[1] > 1.0);
        assert!(f.sample(175.0, 175.0)[1] > 1.0);
    }

    #[test]
    fn slimmer_face_samples_from_further_out() {
        let f = field(FaceSliders { face_width: -100.0, ..Default::default() });
        // Left of centre, the source lies further left (the cheek was pulled in).
        assert!(f.sample(100.0, 150.0)[0] < -0.5);
        assert!(f.sample(200.0, 150.0)[0] > 0.5);
    }

    #[test]
    fn upper_lip_leaves_the_lower_half_alone() {
        let f = field(FaceSliders { upper_lip: 100.0, ..Default::default() });
        assert!(f.sample(150.0, 165.0)[1].abs() > 0.5);
        assert!(f.sample(150.0, 185.0)[1].abs() < 1e-3);
    }

    #[test]
    fn every_slider_at_its_limits_keeps_the_map_unfolded() {
        // At each node the warped source must move forward as the output does (no fold).
        let names = 16;
        for k in 0..names {
            for v in [-100.0, 100.0] {
                let mut s = FaceSliders::default();
                match k {
                    0 => s.eye_size = [v, v],
                    1 => s.eye_height = [v, v],
                    2 => s.eye_width = [v, v],
                    3 => s.eye_tilt = [v, v],
                    4 => s.eye_distance = v,
                    5 => s.nose_height = v,
                    6 => s.nose_width = v,
                    7 => s.smile = v,
                    8 => s.upper_lip = v,
                    9 => s.lower_lip = v,
                    10 => s.mouth_width = v,
                    11 => s.mouth_height = v,
                    12 => s.forehead = v,
                    13 => s.chin_height = v,
                    14 => s.jawline = v,
                    _ => s.face_width = v,
                }
                let f = field(s);
                for j in 0..f.h - 1 {
                    for i in 0..f.w - 1 {
                        let a = f.d[j * f.w + i];
                        let r = f.d[j * f.w + i + 1];
                        let b = f.d[(j + 1) * f.w + i];
                        let c = f.cell as f32;
                        assert!(c + r[0] - a[0] > 0.0 && c + b[1] - a[1] > 0.0, "slider {k} at {v} folds near node {i},{j}");
                    }
                }
            }
        }
    }

    /// Neighbouring nodes never jump apart: a jump shows as a seam (as a one-sided Jawline did
    /// along the mouth line before its edge faded in).
    #[test]
    fn every_slider_at_its_limits_is_seamless() {
        for k in 0..16 {
            for v in [-100.0, 100.0] {
                let mut s = FaceSliders::default();
                let all: [&mut f64; 12] = [
                    &mut s.eye_distance,
                    &mut s.nose_height,
                    &mut s.nose_width,
                    &mut s.smile,
                    &mut s.upper_lip,
                    &mut s.lower_lip,
                    &mut s.mouth_width,
                    &mut s.mouth_height,
                    &mut s.forehead,
                    &mut s.chin_height,
                    &mut s.jawline,
                    &mut s.face_width,
                ];
                match k {
                    0..12 => *all.into_iter().nth(k).unwrap() = v,
                    12 => s.eye_size = [v, v],
                    13 => s.eye_height = [v, v],
                    14 => s.eye_width = [v, v],
                    _ => s.eye_tilt = [v, v],
                }
                let f = field(s);
                for j in 0..f.h - 1 {
                    for i in 0..f.w - 1 {
                        let a = f.d[j * f.w + i];
                        for b in [f.d[j * f.w + i + 1], f.d[(j + 1) * f.w + i]] {
                            let jump = (a[0] - b[0]).hypot(a[1] - b[1]);
                            // Squeezing a feature to half its size moves the source about two pixels
                            // per pixel; the Jawline seam this guards against jumped about 13 px.
                            assert!(jump < 1.5 * f.cell as f32, "slider {k} at {v}: seam of {jump} px at node {i},{j}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn frozen_areas_stay_put() {
        let mut f = LiquifyField::new(Rect::new(0, 0, 300, 320), 2.0);
        f.freeze.iter_mut().for_each(|v| *v = 1.0);
        f.apply_face(&FaceEdit { face: face(), sliders: FaceSliders { face_width: 100.0, ..Default::default() } });
        assert!(f.is_identity());
    }

    #[test]
    fn sliders_out_of_range_are_rejected() {
        let e = FaceEdit { face: face(), sliders: FaceSliders { smile: 150.0, ..Default::default() } };
        assert!(e.validate().is_err());
    }

    #[test]
    fn params_read_back_as_written() {
        let e = FaceEdit { face: face(), sliders: FaceSliders { eye_tilt: [10.0, -5.0], jawline: -20.0, ..Default::default() } };
        let j = serde_json::to_value(e).unwrap();
        assert_eq!(j["eyeTilt"], serde_json::json!([10.0, -5.0]));
        assert_eq!(serde_json::from_value::<FaceEdit>(j).unwrap(), e);
    }
}
