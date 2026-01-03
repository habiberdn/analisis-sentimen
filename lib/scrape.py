import subprocess
import pandas as pd
import io

def scrape_tweets(auth_token : str, keyword :str , limit : int ):
    filename = "cek_kesehatan_gratis.csv"
    
    npx_path = r"C:\Program Files\nodejs\npx.cmd"
    cmd = [
        # gunakan npx path untuk windows, jika linux gunakan "npx"
        npx_path,
        "-y",
        "tweet-harvest@2.6.1",
        "-o", filename,
        "-s", keyword,
        "--tab", "LATEST",
        "-l", str(limit),
        "--token", auth_token,
    ]

    subprocess.run(cmd, check=True)

    file_path = f"tweets-data/{filename}"

    df = pd.read_csv(file_path, delimiter=",")
    return df

def download(firstData, secondData):
    if firstData is None or secondData is None:
        raise ValueError("Kedua file CSV harus diupload")
    df1 = pd.read_csv(firstData)
    df2 = pd.read_csv(secondData)

    combined_df = pd.concat([df1, df2], ignore_index=True)

    # Membuat file palsu di RAM karena DataFrame harus diubah ke file, tapi tanpa menyimpan ke disk
    buffer = io.StringIO()
    combined_df.to_csv(buffer, index=False)
    buffer.seek(0) # posisi pointer berada di akhir, ini berfungsi untuk mengembalikan pointer ke posisi awal

    return buffer.getvalue()
