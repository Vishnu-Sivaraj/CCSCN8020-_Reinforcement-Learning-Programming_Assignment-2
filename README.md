CSCN8020 — Assignment 2
Q-Learning Parameter Analysis (Taxi-v3 Environment)
👨‍🎓 Student Information

Student Name: Vishnu Sivaraj
Course: CSCN8020 – Reinforcement Learning
Instructor: Prof. David Espinosa Carrillo
Institution: Conestoga College
Term: Winter 2026

📌 Project Overview

This assignment investigates the performance of the Q-Learning Reinforcement Learning algorithm using the Taxi-v3 environment from Gymnasium.

The goal is to analyze how different learning rates (α) and exploration rates (ε) affect:

Learning speed

Policy stability

Agent efficiency

Convergence behaviour

Multiple experiments were conducted, evaluated, and compared using performance metrics and visualization plots.

🧠 Environment Description

The Taxi-v3 environment simulates a taxi agent that must:

Navigate a grid world

Pick up a passenger

Deliver the passenger to the correct destination

Environment Properties

State Space: 500 states

Action Space: 6 actions

Reward System

+20 → Successful drop-off

−1 → Step penalty

−10 → Illegal pickup/dropoff

⚙️ Algorithm Used
Q-Learning

Q-Learning is an off-policy Temporal Difference learning algorithm.

Update rule:

𝑄
(
𝑠
,
𝑎
)
=
𝑄
(
𝑠
,
𝑎
)
+
𝛼
[
𝑟
+
𝛾
max
⁡
𝑎
𝑄
(
𝑠
′
,
𝑎
)
−
𝑄
(
𝑠
,
𝑎
)
]
Q(s,a)=Q(s,a)+α[r+γ
a
max
	​

Q(s
′
,a)−Q(s,a)]

Where:

α → Learning rate

γ → Discount factor

ε → Exploration rate (ε-greedy policy)

🧪 Experiments Performed
Learning Rate Experiments

α = 0.001

α = 0.01

α = 0.1

α = 0.2

Exploration Experiments

ε = 0.2

ε = 0.3

Training settings:

Episodes: 5000

Discount factor: γ = 0.9

📊 Evaluation Metrics

The following metrics were used:

Average Return

Average Return (Last 1000 Episodes)

Evaluation Reward

Average Steps per Episode

Success Rate

Training Time

🏆 Best Hyperparameter Combination

Based on experimental results:

Learning Rate (α) = 0.2
Exploration Rate (ε) = 0.1
Discount Factor (γ) = 0.9
Why this works best:

Fast convergence

Highest evaluation reward

Lowest average steps

Stable learning behaviour

Near 100% success rate

📈 Results

Plots showing learning performance are available inside the plots/ folder.

They demonstrate:

Small α → slow learning

Large α → faster convergence

Moderate ε → balanced exploration and exploitation

📂 Project Structure
RL_Assignment2/
│
├── logs/                     # Training logs
├── plots/                    # Learning curves
│
├── assignment2_utils.py      # Provided utility file
├── qlearning_taxi_fixed.py   # Main training script
├── complete_results_fixed.csv
├── Assignment2_Report.pdf
├── requirements.txt
└── README.md
🛠️ Installation

Install required packages:

pip install -r requirements.txt

Dependencies:

numpy

gymnasium

matplotlib

pandas

▶️ How to Run

Run all experiments:

python qlearning_taxi_fixed.py

The script will automatically:

✅ Train the agent
✅ Generate plots
✅ Save logs
✅ Export results table

🧠 Learning Outcomes

This project demonstrates:

Reinforcement learning fundamentals

Exploration vs exploitation trade-off

Hyperparameter tuning

Performance evaluation of RL agents

📄 License

Academic use only — Conestoga College coursework.