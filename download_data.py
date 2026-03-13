import urllib.request
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = "https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt"
with urllib.request.urlopen(url, context=ctx) as response:
    with open("data/shakespeare.txt", "wb") as f:
        f.write(response.read())

print("Download complete!")