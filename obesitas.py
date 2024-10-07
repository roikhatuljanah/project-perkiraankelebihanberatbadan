import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# Call st.set_page_config as the very first Streamlit command
st.set_page_config(
    page_title="Aplikasi Prediksi Kelebihan Berat Badan",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None
)

# Define the sidebar menu
with st.sidebar:
    selected = option_menu("Menu Utama", ["Dashboard", "Visualisasi Data", "Perhitungan"], icons=['house', 'pie-chart'], menu_icon="cast", default_index=0)

# Define a function for the "Dashboard" page
def dashboard_page():
    st.markdown("---")
    st.title('Aplikasi Prediksi Kelebihan Berat Badan')
    st.write("")
    st.markdown("---")

# Define a function for the "Visualisasi Data" page
def data_visualization_page():
    st.title('')
    
    # Baca berkas CSV ke dalam DataFrame
    df = pd.read_csv('Training_ob.csv')
    
    # Tampilkan informasi dasar tentang data
    st.subheader("Ikhtisar Data")
    st.write("Jumlah baris:", df.shape[0])
    st.write("Jumlah kolom:", df.shape[1])
    
    # Tampilkan beberapa baris pertama data
    st.subheader("Pratinjau Data")
    st.dataframe(df.head())
    
    # Analisis data dan visualisasi dapat ditambahkan di sini
    
    # Contoh: Gambar diagram batang dari kolom tertentu
    st.subheader("Diagram Batang")
    st.write(" grafik batang yang menampilkan sebaran data dalam kolom yang dipilih.")
    selected_column = st.selectbox("Pilih kolom untuk diagram batang", df.columns, key="bar_chart_selectbox")
    st.bar_chart(df[selected_column])

    # Contoh: Tampilkan statistik ringkas
    st.subheader("Statistik Ringkas")
    st.write(" ringkasan statistik tentang data numerik dalam dataset. Statistik ini mencakup informasi seperti jumlah data, rata-rata, deviasi standar, nilai minimum, dan nilai maksimum.")
    st.write(df.describe(), key="summary_stats")
    
    if st.checkbox("Visualisasi data"):
        # Create Logistic Regression plots for five numeric columns
        st.subheader("Logistic Regression Plots")
        st.write("Pilih lima kolom numerik untuk plot regresi logistik.")
        selected_columns_lr = st.multiselect("Pilih kolom numerik", df.select_dtypes(include=['int64', 'float64']).columns, default=[])

        if selected_columns_lr:
            for column in selected_columns_lr:
                X = df[[column]]
                y = df['thal']  

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                model = LogisticRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                fig, ax = plt.subplots()
                ax.scatter(X_test, y_test, color='blue', label='Actual', alpha=0.5)
                ax.scatter(X_test, y_pred, color='red', label='Predicted', alpha=0.5)
                ax.legend()
                st.write(f"Logistic Regression Plot for {column}")
                st.pyplot(fig)

                st.subheader("Confusion Matrix")
                st.write("Visualize the confusion matrix.")
                cm = confusion_matrix(y_test, y_pred)
                labels = ['Negative', 'Positive']
                sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=labels, yticklabels=labels)
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                st.pyplot()

        naive_bayes_predictions = [0, 1, 1, 0, 1, 0, 1]  # Contoh prediksi
        
        # Buat DataFrame Pandas untuk menyimpan prediksi
        st.write(" ringkasan statistik tentang data numerik dalam dataset. Statistik ini mencakup informasi seperti jumlah data, rata-rata, deviasi standar, nilai minimum, dan nilai maksimum.")
        prediction_df = pd.DataFrame({'Prediksi': naive_bayes_predictions})
        
        # Gambar diagram batang untuk visualisasi prediksi
        st.write("grafik batang yang menunjukkan hasil prediksi")
        st.bar_chart(prediction_df)


        # Buat plot pencocokan pasangan untuk visualisasi hubungan pasangan kolom numerik.
        st.subheader("Plot Pencocokan Pasangan")
        st.write("kumpulan plot pencocokan pasangan antara kolom numerik dalam data. Ini membantu pengguna untuk melihat hubungan antara setiap pasang atribut.")
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        sns.pairplot(df[numeric_cols])
        st.pyplot()

        # Izinkan pengguna memilih kolom numerik dan tampilkan diagram kotak dari nilainya.
        st.subheader("Diagram Kotak")
        st.write("diagram kotak yang menunjukkan distribusi data dalam kolom numerik yang dipilih. Diagram kotak ini menyoroti statistik seperti median, kuartil, dan pencilan.")
        selected_column_box = st.selectbox("Pilih kolom numerik untuk diagram kotak", df.select_dtypes(include=['int64', 'float64']).columns, key="box_selectbox")
        sns.boxplot(data=df, x=selected_column_box)
        st.pyplot()

        # Buat diagram batang dari kolom kategoris untuk visualisasi distribusi kategori.
        st.subheader("Diagram Batang (Count Plot)")
        st.write(" diagram batang yang menunjukkan sebaran kategori dalam kolom kategoris yang dipilih.")
        selected_column_count = st.selectbox("Pilih kolom kategoris untuk diagram batang", df.select_dtypes(include=['object']).columns, key="count_selectbox")
        sns.countplot(data=df, x=selected_column_count)
        st.pyplot()

        # Izinkan pengguna memilih dua kolom numerik dan tampilkan plot pencocokan pasangan dari nilai keduanya.
        st.subheader("Plot Pencocokan (Scatter Plot)")
        st.write(" plot pencocokan pasangan antara dua kolom numerik yang dipilih oleh pengguna. Ini membantu pengguna memahami hubungan antara dua atribut dalam bentuk plot pencocokan pasangan.")
        x_column = st.selectbox("Pilih kolom sumbu X untuk plot pencocokan pasangan", df.select_dtypes(include=['int64', 'float64']).columns, key="x_selectbox")
        y_column = st.selectbox("Pilih kolom sumbu Y untuk plot pencocokan pasangan", df.select_dtypes(include=['int64', 'float64']).columns, key="y_selectbox")
        plt.scatter(df[x_column], df[y_column])
        st.pyplot()

        st.subheader("Heatmap Korelasi")
        numerical_columns = df.select_dtypes(include=['int64', 'float64'])
        sns.heatmap(numerical_columns.corr(), annot=True, cmap='coolwarm')
        st.pyplot()

# Define a function for the "Perhitungan" page
def Perhitungan_page():
    st.title('Perhitungan Kelebihan Berat Badan')
    
    model = pickle.load(open('obesitas.sav', 'rb'))

    # Bidang masukan untuk pengguna memasukkan data
    Tinggi_Badan = st.text_input('Tinggi Badan')
    Berat_Badan = st.text_input('Berat Badan')

    prediction = ''

    # Tangani klik tombol untuk membuat prediksi
    if st.button('TEST PREDIKSI'):
        # Convert input values to numeric
        
        Tinggi_Badan = int(Tinggi_Badan)
        Berat_Badan = int(Berat_Badan)

        data_baru = np.array([Tinggi_Badan, Berat_Badan]).reshape(1, -1)

        prediction = model.predict(data_baru)
        
        # Tampilkan hasil prediksi
        st.success(f'Hasil Prediksi: {prediction}')

# Tergantung pada halaman yang dipilih, tampilkan konten yang sesuai
if selected == "Dashboard":
    dashboard_page()
elif selected == "Visualisasi Data":
    data_visualization_page()
elif selected == "Perhitungan":
    Perhitungan_page()
