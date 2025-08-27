from flask import Flask, request, render_template
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        data = pd.read_csv(file, encoding="latin1")

        # Compute the Amount (Monetary) column for RFM
        data['Amount'] = data['Quantity'] * data['UnitPrice']

        # Calculate RFM table
        data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], format="%d-%m-%Y %H:%M")

        now = data['InvoiceDate'].max()
        rfm = data.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (now - x.max()).days,  # Recency
            'InvoiceNo': 'count',                          # Frequency
            'Amount': 'sum'                                # Monetary
        }).reset_index()
        rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

        # Dynamically set cluster count: min(3, number of customers)
        num_customers = rfm.shape[0]
        num_clusters = min(3, num_customers)
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        rfm['Cluster_id'] = kmeans.fit_predict(X_scaled)

        # Create cluster plots
        plt.figure(figsize=(15, 5))
        for fid, (feat, ylabel) in enumerate(zip(['Monetary', 'Frequency', 'Recency'], ['Amount', 'Frequency', 'Recency'])):
            plt.subplot(1, 3, fid+1)
            for cid in sorted(rfm['Cluster_id'].unique()):
                subset = rfm[rfm['Cluster_id'] == cid]
                plt.scatter([cid]*len(subset), subset[feat], label=f"Cluster {cid}", alpha=0.7)
            plt.xlabel("cluster_id")
            plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig('static/result.png')
        plt.close()
        print("Rendering result page with image:", 'static/result.png')
        return render_template('result.html', img_file='static/result.png')



    return render_template('upload.html')



if __name__ == '__main__':
    app.run(debug=True)
