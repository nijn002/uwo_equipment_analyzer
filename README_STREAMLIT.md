# GVO Equipment Analyzer - Streamlit Version

A web-based tool for analyzing and finding optimal equipment combinations in GVO.

## Deploying to Streamlit Cloud (Private Data)

1. First, convert your CSV file to base64. You can use this Python script:
   ```python
   import base64
   
   with open('equipment.csv', 'rb') as f:
       csv_bytes = f.read()
       csv_base64 = base64.b64encode(csv_bytes).decode('utf-8')
       print(csv_base64)
   ```

2. Create a GitHub repository and push your code (WITHOUT the CSV file):
   ```bash
   git init
   git add app.py requirements.txt README_STREAMLIT.md
   git commit -m "Initial commit"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

3. Go to [Streamlit Cloud](https://share.streamlit.io/) and sign in with your GitHub account.

4. Click "New app" and select your repository.

5. Before deploying, set up your secrets:
   - Click on "Advanced settings"
   - Under "Secrets", add these two keys:
     - `password`: Your chosen password for accessing the app
     - `equipment_csv`: The base64-encoded CSV data you generated in step 1

6. Configure the deployment:
   - Main file path: `app.py`
   - Python version: 3.7 or newer

7. Click "Deploy"

Your app will be available at a URL like: `https://share.streamlit.io/yourusername/yourrepo/main/app.py`

## Security Features

- Password protection prevents unauthorized access
- CSV data is stored securely in Streamlit's secrets management
- Data never appears in the GitHub repository
- Each user session is isolated

## Local Development

1. Create a `.streamlit/secrets.toml` file in your project directory:
   ```toml
   password = "your_chosen_password"
   equipment_csv = "your_base64_encoded_csv"
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app locally:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Enter the password to access the app
2. Use the interface to:
   - Select equipment using the dropdowns
   - View total stats and skill boosts in real-time
   - Find optimal equipment combinations based on requirements

## Notes

- The app will remember your session state while you're using it
- Equipment combinations are sorted by their effectiveness
- The "Include not-released" option shows equipment that's only available in the Japanese version 