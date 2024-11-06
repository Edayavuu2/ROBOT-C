import os

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)  # Kayıt dizini oluştur
    import train  # Eğitim işlemini başlat

