
# Handwritten Digit Recognition (0–9) System

This project is a handwritten digit recognition system that uses a simple neural network built from scratch using only `numpy` and basic math (no ML libraries like TensorFlow or PyTorch). It uses the MNIST dataset and predicts digits (0–9) based on an image uploaded by the user via a frontend interface.

---

## 📁 Project Structure

```
handwriting-recognition/
├── frontend/
│   └── placeholder.md        # Placeholder for frontend files
├── backend/
│   ├── notebooks/
│   │   └── Handwritten_Recognition.ipynb  # Jupyter notebook with model training and testing
│   ├── dataset/              # Contains MNIST training and testing CSV files
│   └── env/                  # Python virtual environment (not versioned)
├── .gitignore
└── README.md
```

---

## 💻 Setup Instructions

### ✅ Prerequisites

- Python 3.8+
- `pip`
- Kaggle account + `kaggle.json` API token
- Git (optional, for cloning)

---

### 🧪 Setting Up the Backend

#### 1. Clone the repository (or download manually)

```bash
git clone https://github.com/your-username/handwriting-recognition.git
cd handwriting-recognition/backend
```

#### 2. Create a virtual environment

**Linux/macOS:**

```bash
python3 -m venv env
source env/bin/activate
```

**Windows (CMD):**

```cmd
python -m venv env
env\Scripts\activate
```

**Windows (PowerShell):**

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
```

#### 3. Install dependencies

```bash
pip install -r requirements.txt
```

#### 4. Get the MNIST CSV dataset from Kaggle
- Download `kaggle.json` file from your kaggle account:
	- Go to your account, Scroll to API section and Click Expire API Token to remove previous tokens
	- Click on Create New API Token - It will download kaggle.json file on your machine.
	
- Place your `kaggle.json` in the `backend` folder.
- **For windows users:** While being in the backend folder, right click on any blank area and click on `Open in Terminal`.
- Then run the following:

> On **Windows**, instead of `cp`, `chmod`, and `mv`, use these:
```cmd
copy kaggle.json %USERPROFILE%\.kaggle\
kaggle datasets download -d oddrationale/mnist-in-csv
tar -xf mnist-in-csv.zip
mkdir dataset
move mnist_train.csv dataset\
move mnist_test.csv dataset\
```

> On Linux/MacOS, run these:
```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d oddrationale/mnist-in-csv
unzip mnist-in-csv.zip
mkdir -p dataset
mv mnist_train.csv mnist_test.csv dataset/
```

---

### 📓 Run the Notebook

Once everything is set up:

```bash
jupyter notebook
```

Then navigate to `notebooks/Handwritten_Recognition.ipynb` and run all cells to train/test the model.

## 👨‍💻 Author

Made with ❤️ by **Saiyed Shizain**. Inspired by classic deep learning exercises.
