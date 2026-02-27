# 🚕 CSCN8020 – Reinforcement Learning Programming  
## Assignment 2 – Q-Learning Parameter Analysis (Taxi-v3)

---

## 👨‍🎓 Student Information
**Student:** Vishnu Sivaraj  
**Course:** CSCN8020 – Reinforcement Learning Programming  
**Instructor:** Prof. David Espinosa Carrillo  
**Institution:** Conestoga College  
**Term:** Winter 2026  

---

## 📌 Project Overview

This project analyzes the performance of the **Q-Learning Reinforcement Learning algorithm** using the **Taxi-v3 environment** from Gymnasium.

The objective is to understand how different **learning rates (α)** and **exploration rates (ε)** influence:

- Learning speed
- Convergence stability
- Policy efficiency
- Agent performance

Multiple experiments were conducted and evaluated using quantitative metrics and visualization plots.

---

## 🧠 Environment Description

Taxi-v3 simulates a taxi agent operating in a grid world.

### Task
1. Navigate environment
2. Pick up passenger
3. Drop passenger at destination

### Environment Properties
- **States:** 500
- **Actions:** 6
- **Rewards**
  - +20 → Successful drop-off
  - −1 → Each movement step
  - −10 → Illegal pickup/dropoff

---

## ⚙️ Algorithm

### Q-Learning (Off-Policy TD Control)

Update rule:

\[
Q(s,a) = Q(s,a) + \alpha [r + \gamma \max_a Q(s',a) - Q(s,a)]
\]

Where:

- α → Learning Rate  
- γ → Discount Factor  
- ε → Exploration Rate (ε-greedy)

---

## 🧪 Experiments Conducted

### Learning Rate Experiments
- α = 0.001
- α = 0.01
- α = 0.1
- α = 0.2

### Exploration Experiments
- ε = 0.2
- ε = 0.3

Training configuration:

- Episodes: **5000**
- Discount factor: **γ = 0.9**

---

## 📊 Evaluation Metrics

The agent performance was evaluated using:

- Average Return
- Average Return (Last 1000 Episodes)
- Evaluation Reward
- Average Steps
- Success Rate
- Training Time

---

## 🏆 Best Hyperparameter Combination
α = 0.2
ε = 0.1
γ = 0.9


### Why this works best
✅ Fast convergence  
✅ Highest evaluation reward  
✅ Lowest number of steps  
✅ Stable learning behaviour  
✅ Near 100% success rate  

---

## 📈 Results Visualization

Learning curves for all experiments are available in:
plots/


They demonstrate:

- Small learning rate → slow learning
- High learning rate → faster convergence
- Balanced exploration → optimal performance

---

## 📂 Repository Structure

RL_Assignment2/
│
├── plots/ # Training graphs
├── logs/ # Experiment logs
│
├── assignment2_utils.py # Instructor provided utilities
├── qlearning_taxi_fixed.py # Main training script
├── complete_results_fixed.csv # Experiment results
├── Assignment2_Report.pdf # Final report
├── requirements.txt
└── README.md


---

## 🛠 Installation

Install dependencies:

```bash

pip install -r requirements.txt

Required packages:

numpy

gymnasium

matplotlib

pandas

------

▶️ Run Experiments

python qlearning_taxi_fixed.py

The script will:

✅ Train agent
✅ Run all experiments
✅ Generate plots
✅ Save logs
✅ Export results table

-----

🧠 Learning Outcomes

This assignment demonstrates:

Reinforcement Learning fundamentals

Exploration vs Exploitation trade-off

Hyperparameter tuning

Experimental evaluation of RL agents

----

📄 License

Academic use only — Conestoga College Coursework.