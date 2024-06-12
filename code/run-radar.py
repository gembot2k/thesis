import os
from time import time

import lib.uRAD_USB_SDK11 as uRAD_USB_SDK11
import serial

duration = 60
usb_communication = True

# input parameters
mode = 2
f0 = 5
BW = 240
Ns = 200
Ntar = 3
Rmax = 100
MTI = 0
Mth = 0
Alpha = 10
distance_true = False
velocity_true = False
SNR_true = False
I_true = True
Q_true = True
movement_true = False

# Serial Port configuration
ser = serial.Serial()
ser.port = "/dev/cu.usbmodem101"
ser.baudrate = 1e6  # type: ignore

# Other serial parameters
ser.bytesize = serial.EIGHTBITS
ser.parity = serial.PARITY_NONE
ser.stopbits = serial.STOPBITS_ONE
ser.timeout = 1


# Method to correctly turn OFF and close uRAD
def closeProgram():
    # switch OFF uRAD
    return_code = uRAD_USB_SDK11.turnOFF(ser)
    if return_code != 0:
        exit()


# Open serial port
try:
    ser.open()
except Exception:
    closeProgram()

# switch ON uRAD
return_code = uRAD_USB_SDK11.turnON(ser)
if return_code != 0:
    closeProgram()

# loadConfiguration uRAD
return_code = uRAD_USB_SDK11.loadConfiguration(
    ser,
    mode,
    f0,
    BW,
    Ns,
    Ntar,
    Rmax,
    MTI,
    Mth,
    Alpha,
    distance_true,
    velocity_true,
    SNR_true,
    I_true,
    Q_true,
    movement_true,
)
if return_code != 0:
    closeProgram()

resultsFileName = "IQ.txt"
fileResults = open(resultsFileName, "a")
iterations = 0
t_0 = time()

# infinite detection loop
while True:
    return_code, results, raw_results = uRAD_USB_SDK11.detection(ser)
    if return_code != 0:
        closeProgram()

    # Extract results from outputs
    I = raw_results[0]  # noqa: E741
    Q = raw_results[1]

    t_i = time()

    IQ_string = ""
    for index in range(len(I)):
        IQ_string += "%d " % I[index]
    for index in range(len(Q)):
        IQ_string += "%d " % Q[index]

    fileResults.write(IQ_string + "%1.3f\n" % t_i)

    iterations += 1

    if iterations > 100:
        print("Fs %1.2f Hz" % (iterations / (t_i - t_0)))

    if (t_i - t_0) >= duration:
        Fs = round(iterations / (t_i - t_0), 2)
        break

closeProgram()

WDIR = "./radar/"
ID_sample = str(round(time()))
IQ_file = "IQ_" + ID_sample + ".txt"
os.rename(resultsFileName, WDIR + IQ_file)

RRm = float(input("Enter RR guided: "))

dataset = WDIR + "dataset.txt"
GT_data = " ".join([IQ_file, str(duration), str(Fs), str(RRm)])
with open(dataset, "a") as file:
    file.write(GT_data + "\n")
