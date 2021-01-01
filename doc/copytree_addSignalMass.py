import ROOT
from ROOT import *
import math
from array import array
from scipy import stats 

doVal = False
#use this for assigning masses to signal:
sigMass = 700 #or any other mass
#MassForBkg = -1

if doVal:
  process_name = "TTBar_c16d"
else:
  process_name = "LQ_{}GeV_c16d".format(int(sigMass))


oldfile = TFile.Open("dumpNtuples/"+process_name+".root")
oldtree = oldfile.NOMINAL

if doVal:
  process_name = process_name + "_" + str(sigMass) + "GeV"

newfile = TFile("dumpNtuples/"+process_name+"_WithSigMassAndLabel.root","RECREATE")
newtree = oldtree.CloneTree(0)

events = oldtree.GetEntries()


MassArray = array("i", [0])
signal_mass = newtree.Branch( "signal_mass" , MassArray, 'signal_mass/I' ) 
LabelArray = array("i", [0])
signal_label = newtree.Branch( "signal_label" , LabelArray, 'signal_label/I' ) 


#use this for assigning masses to background:
#massPoints = (200,300,400,500,600,700,800,900,1000,1200,1500)
massPoints = (400,600,800,1000,1500)
#probability distribution:
weights = (2974.0/20814,4203.0/20814,4760.0/20814,4682.0/20814,4195.0/20814) #number signal events per mass point, divided by the total number of signal events - c16a
#(or you can normalize signal samples and use a flat probability distribution here)

distribution = stats.rv_discrete(name='distribution', values=(massPoints, weights))

for i in range(events):
  oldtree.GetEntry(i)

  mass = 0
  label = 0
  if "LQ" not in process_name:
    #bkg:
    label = 0
    mass = distribution.rvs(size=1)
    #mass = MassForBkg
    if doVal:
      mass = sigMass
  elif "LQ" in process_name:
    #sig:
    label = 1
    mass = sigMass
  MassArray[0] = int(mass)
  print(MassArray[0])
  LabelArray[0] = int(label)
  print(LabelArray[0])
  newtree.Fill()



newfile.Write()
newfile.Close()


