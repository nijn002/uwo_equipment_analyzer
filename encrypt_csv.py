import base64

# Read the CSV file and convert to base64
with open('equipment.csv', 'rb') as f:
    csv_bytes = f.read()
    csv_base64 = base64.b64encode(csv_bytes).decode('utf-8')

# Save the base64 string to a file for easy copying
with open('equipment_base64.txt', 'w') as f:
    f.write(csv_base64)

print("CSV has been converted to base64 and saved to 'equipment_base64.txt'")
print("Use this value in your Streamlit secrets")