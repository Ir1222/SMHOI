# M6.5 protocol and immutable baseline

Read V3 plan, REPORT.md, VALIDATION.md and complete current asset manifest. No applicable AGENTS.md found. M6 history is frozen by SHA256; new outputs go only under m6_5. Existing source data and validated robot arrays are read-only. No M7–M9 or learning.

A: decompose point error into box-wall normal/tangential components, fit diagnostic constant EEF/target alternatives, compare original/HDMI geometry and bounded shape changes. Coordinate/orientation ablations are diagnostic counterfactuals, not proposed robot trajectory edits. Target relabeling alone cannot change penetration. Limited object-pose feasibility search may test geometry with the robot fixed, without producing or replacing reference motion.

B: recompute all 840 frames and actual convex polygon footprint intersection; compare support Z offsets 0, +4, +7 and +10 mm and determine feasible static-height intervals. Distinguish infinite-plane separation from footprint/pair collision. Use fixed supports and documented geometry, not guessed dimensions.

C: use the Reproduced get_g1_robot_cfg + Entity actuator construction, original joint position setpoint interface, gains/effort limits/armature. Reference joint targets at 50 Hz, MuJoCo 200 Hz. No reference velocity feedforward (original position actuator damps toward zero joint velocity), no root wrench, no box control, no weld, no post-reset state writes. Controlled evaluation disables action noise/randomization; feet remain native, nominal evaluation friction 1.0. Free robot root and free seven-part box. Short paired windows: first contact/pickup, carry, placement/release, and final placement. Box-only settle tests isolate support stability from PD balance failure. Ground plane is present. Measured forces/penetration/slip and full object/root traces required; tracking is not task success.

Candidates are documented before dynamics. Strict original-reference geometric validity and actual released-object stability are separate verdicts. A short PD failure cannot prove no learned controller could succeed, and a terminal settle success cannot prove successful delivery. No hidden correction or gate relaxation.
