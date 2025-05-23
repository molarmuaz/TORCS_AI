# AI Racing Driver 🏁

This project was built as part of an AI course — the goal was to train an end-to-end neural network to drive a racing car in the TORCS simulator.

I gathered data from in-game sessions, experimented with different inputs and output mappings, played with sensor sensitivity, tuned model weights, and tested driving performance under various conditions. The result: an AI that can steer, throttle, brake, and shift gears without any rule-based logic.

---

## 🔧 Technologies Used

* **Python**
* **PyTorch**
* **scikit-learn**
* **Pandas**
* **NumPy**

---

## 📊 Sensors Used

* Track edge distances
* Speed (X, Y, Z)
* RPM
* Gear
* Angle to track axis
* Opponent proximity (optional)

---

## 🧠 How It Works

1. **Data Collection**: Logged sensor values + driving actions from human gameplay
2. **Model Design**: Neural net trained to predict control actions from sensor input
3. **Experimentation**: Tuned inputs, outputs, weights, dropout, normalization, and loss function
4. **Execution**:

   * Train with:

     ```bash
     python train_model.py
     ```
   * Run AI driver with:

     ```bash
     python ai_driver.py
     ```

---

## 📦 Requirements

Install dependencies via:

```bash
pip install -r requirements.txt
```

---

## 🔗 GitHub Repo

[github.com/molarmuaz/TORCS\_AI](https://github.com/molarmuaz/TORCS_AI)

---

Let me know if you want a one-line project blurb for your resume or LinkedIn too.
