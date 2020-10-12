file = open("FC_label.txt");
lines=file.readlines()
label=[]
for line in lines:
        label.append(line.split("\n"))
print (label)