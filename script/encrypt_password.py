from cryptography.fernet import Fernet
from statics import ROOT_DIR

key_path = ROOT_DIR + "/script/db/refKey.txt"
f = open(key_path, "rb")
key = f.read()
f.close()
refKey = Fernet(key)
new_pass = input("new password to encrypt: ")
mypwdbyt = bytes(new_pass, 'utf-8')
encryptedPWD = refKey.encrypt(mypwdbyt)
print(encryptedPWD.decode('utf-8'))
