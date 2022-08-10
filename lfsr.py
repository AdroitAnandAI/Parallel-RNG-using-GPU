start_state = 1 << 23 | 1
lfsr = start_state
period = 0


file = open('lfsrout.txt', 'w')

while True:
    #taps: 16 15 13 4; feedback polynomial: x^16 + x^15 + x^13 + x^4 + 1
    # taps: x^{24}+x^{23}+x^{22}+x^{17}+1
    bit = (lfsr ^ (lfsr >> 1) ^ (lfsr >> 2) ^ (lfsr >> 7)) & 1
    lfsr = (lfsr >> 1) | (bit << 23)
    file.write(str(lfsr)+"\n")
    period += 1
    if (lfsr == start_state):
        print(period)
        file.close()
        break


