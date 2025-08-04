from Iris import *
sepallength=float(input('Enter the sepallength: '))
sepalwidth=float(input('Enter the sepalwidth: '))
petallength=float(input('Enter the petallength: '))
petalwidth=float(input('Enter the petalwidth: '))
predictor=supportvector.predict([[sepallength,sepalwidth,petallength,petalwidth]])
species_name = le.inverse_transform(predictor)[0]
print("species name is: ",species_name)
