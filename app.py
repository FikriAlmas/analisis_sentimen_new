import streamlit as st
import time
from predict import predict

def main():
    # Header
    st.title('🎥 Prediksi Sentimen Ulasan Martabak Kubang Hayuda')
    st.markdown(
        """
        Masukkan ulasan tentang bioskop untuk mengetahui sentimen dan aspek terkait.
        Analisis ini menggunakan **model machine learning** untuk memprediksi hasilnya.
        """
    )

    # Input ulasan
    text = st.text_area('📝 Masukkan Teks Ulasan:', 
                        'Contoh: makanan enak', 
                        height=150)
    
    if st.button('🔍 Mulai Analisis'):
        if text.strip() == '':
            st.error('Teks ulasan tidak boleh kosong. Silakan masukkan teks yang valid.', icon="🚨")
        else:
            with st.spinner('🔄 Memproses ulasan, harap tunggu...'):
                time.sleep(2)
                sentimen, aspek = predict(text)
                st.success('Analisis berhasil!')

            # Menampilkan hasil 
            st.subheader('📊 Hasil Analisis:')
            st.markdown(f"**Sentimen:** {sentimen}")
            st.markdown(f"**Aspek yang Disorot:** {aspek}")
    else:
        st.info('Masukkan ulasan lalu tekan tombol **Mulai Analisis** untuk melihat hasil.')

# Jalankan aplikasi
if __name__ == '__main__':
    main()
