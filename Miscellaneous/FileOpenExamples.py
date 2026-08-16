'''
Hasan Helal
This program reads inputs from a textfile and configures the errors
'''

while True:
    try:
        fname = input("Please enter the file name: ")
        inFile = open(fname, "r")
        break
    except IOError:
        print("Could not open input file")

reportf = open("reportError.txt", 'w')

count = 0
countError = 0
for line in inFile:
    line = line.rstrip()
    
    if(line != ""):
        count += 1
    
    if("error" in line):
        print(line)
        reportf.write(line + "\n")
        countError += 1
    
    elif("Error" in line):
        print(line)
        reportf.write(line + "\n")
        countError += 1
    
    elif("ERROR" in line):
        print(line)
        reportf.write(line + "\n")
        countError += 1

print("Count:", count)
print("Error Count:", countError)

reportf.write("Count: " + str(count) + "\n")
reportf.write("Error Count: " + str(countError) + "\n")

inFile.close()
reportf.close()

"""
Example:
[Sun Mar  7 21:16:17 2018] [error] [client 24.70.56.49] File does not exist: /home/httpd/twiki/view/Main/WebHome
[Mon Mar  8 07:27:36 2018] [error] [client 61.9.4.61] File does not exist: /usr/local/apache/htdocs/_vti_bin/owssvr.dll
[Mon Mar  8 07:27:37 2018] [error] [client 61.9.4.61] File does not exist: /usr/local/apache/htdocs/MSOffice/cltreq.asp
[Thu Mar 11 02:27:34 2018] [error] [client 200.174.151.3] File does not exist: /usr/local/mailman/archives/public/cipg/2018-november.txt
[Thu Mar 11 07:39:29 2018] [error] [client 140.113.179.131] File does not exist: /usr/local/apache/htdocs/M83A
Count: 108
Error Count: 5
"""