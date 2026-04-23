# Semester Project Proposal: Baseball Identification & Tracking
**Course:** ECON 8310 - Business Forecasting
**Team Members:** Sadie Lamplot, Adam Nutt, Alyssa Miller

---

## 1. Problem Statement
Our team has been tasked to develop a computer vision solution capable of identifying and tracking baseballs within training videos. The primary objective is to produce highly accurate, tight bounding boxes around the moving baseball in every frame to provide the coach with actionable data for player development.

## 2. Proposed Approach
To address the complexity of identifying a small, fast-moving object, our team will implement a multi-stage pipeline focused on neural networks using **PyTorch**. While we will build our architecture from the ground up to ensure it is tailored to this specific environment, we plan to utilize **transfer learning**. This allows us to leverage pre-trained weights to give our model a foundational understanding of shapes before fine-tuning it on our custom-labeled baseball data.

### Technical Workflow:
* **Collaborative Labeling:** We will utilize **CVAT** to manually annotate training videos, ensuring a high-quality shared dataset for the class.
* **Pre-processing & Motion Isolation:** To combat "noisy" backgrounds, we will implement **Frame Differencing**. This technique allows us to subtract static background elements and highlight only the pixels in motion, significantly reducing the "search area" for our mode.
* **Model Training:** Using the labeled data, we will train our PyTorch model to optimize for **Intersection over Union (IoU)**, ensuring the bounding boxes are as precise as possible.

## 3. Data Request and Justification
To improve the robustness of our model, we request additional footage captured under the following conditions:

| Condition | Justification |
| :--- | :--- |
| **High-Contrast Backgrounds** | Videos filmed against dark screens or green turf to help the model distinguish the white baseball more easily. |
| **Varied Camera Angles** | Footage from "catcher's view" and "side-profile" to help the model recognize the ball's scale as it moves toward or across the lens. |
| **High Frame Rate (FPS)** | Minimal motion blur is required to ensure the "tightest" possible bounding boxes, as blurry streaks make precise localization difficult. |

## 4. Conclusion
By combining classical motion-detection techniques like frame differencing with modern neural networks, our group aims to deliver a professional-grade forecasting tool that turns "messy" training footage into a streamlined diagnostic resource for the coaching staff.
