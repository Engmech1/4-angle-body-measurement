# Standard Anthropometric Capture Protocol (PROTOCOL.md)

This protocol establishes the standardized operating procedure (SOP) to minimize biological and postural variance in accordance with international anthropometric standards (ISAK / ISO 7250).

---

## 1. Environmental & Hardware Setup

1. **Camera Position**:
   - Tripod height set to the subject's navel level (approx. 95–105 cm from the floor).
   - Camera lens leveled horizontally (pitch and roll $\le 1.0^\circ$).
   - Distance from lens to subject rotation center: $2.20\text{ m} \pm 0.10\text{ m}$.

2. **Calibration Reference**:
   - Place the ArUco calibration board in the subject's coronal plane (at the turntable center or toe-alignment line).
   - Ensure the ArUco board is free of specular glare.

3. **Illumination**:
   - Diffuse softbox or ring light placed behind or beside the camera to illuminate the subject uniformly with high silhouette contrast against the background.

---

## 2. Subject Preparation & Attire

1. **Clothing**:
   - Form-fitting athletic attire (e.g. compression shorts, sports bra) or minimal clothing. Loose clothing creates artificial silhouette expansion.
   - Hair tied up above the neck.

2. **Physiological State**:
   - Measurements should be taken at the same time of day (preferably morning, fasted state).
   - Record time since last meal and hydration status.

---

## 3. Pose & Respiration Guidelines

1. **Stance**:
   - Feet positioned at shoulder width, parallel, centered on the turntable mat.
   - Weight distributed equally between both feet (50/50).

2. **Arm Abduction**:
   - Arms abducted laterally at $30^\circ \pm 5^\circ$ from the torso with hands open and palms facing inwards.
   - This prevents forearm/arm occlusion of the lateral waist flanks.

3. **Breathing Phase**:
   - **End-tidal normal exhalation**: The subject breathes normally and capture is triggered at the end of a quiet exhale (NOT forced exhalation).

---

## 4. Capture Gating Logic

The capture HUD automatically evaluates and **blocks capture** if:
- Arm abduction angle $< 20^\circ$ or $> 40^\circ$.
- Spine lateral tilt $> 2.0^\circ$.
- Distance to camera exceeds $2.2\text{ m} \pm 0.25\text{ m}$.
- Yaw angle deviates from target angle bin by $> 8^\circ$.
- Subject movement/sway velocity exceeds $3.0\text{ cm/s}$.
