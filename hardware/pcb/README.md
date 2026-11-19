# AFIB Carrier Rev B — Prototype Fabrication Candidate

This project is the small-batch prototype candidate derived from the uploaded
Rev A archive. Open `afib_carrier.kicad_pro` in KiCad 10.

## Release checks

- KiCad 10.0.6 PCB DRC: **0 violations**
- Unconnected PCB items: **0**
- Schematic/PCB parity issues: **0**
- Schematic ERC: **0 violations**
- Minimum track: 0.25 mm
- Minimum via drill: 0.30 mm
- Minimum configured copper clearance: 0.20 mm
- Minimum configured copper-to-edge clearance: 0.50 mm
- ESP32 antenna keepout coordinates checked after save/reload
- Battery B− remains separate from charger OUT−/system GND

## Board specification

- 90.00 × 80.00 mm, 1.60 mm FR-4
- Two copper layers, 1 oz copper recommended
- R2 mm corners
- Four Ø3.20 mm NPTH mounting holes at (4,4), (86,4), (4,76),
  (86,76) mm from the top-left drawing origin
- Lead-free HASL is sufficient; ENIG is optional
- Order 2–5 boards first and electrically test one fully before a larger batch

## BO-10186 footprint used

The supplied listing screenshots state **2.54 mm pitch** and **20 × 15 × 5 mm**.
The carrier therefore uses a normal 1×4 2.54 mm through-hole pattern with
1.0 mm plated drills and a conservative 20 × 15 mm reserved body envelope.
The header centreline is placed 1.27 mm from the reserved left edge, matching
the product photographs. The complete envelope is inside the carrier outline.

Top-to-bottom at U3:

| Pad | Module marking | Net | Board XY, mm |
|---|---|---|---|
| 1 | VOUT | +3V3 | 69.47, 57.00 |
| 2 | GND | GND | 69.47, 59.54 |
| 3 | GND | GND | 69.47, 62.08 |
| 4 | VIN | REG_IN_POS | 69.47, 64.62 |

Pad 1 is square. Fit the module with its component side upward and body extending
to the right of the header. If the supplied module differs from the listing,
the same four labelled carrier pads can be connected using short insulated wires;
do not force a mismatched module into a rigid socket.

## MAX30003 socket

The eight-pin digital header uses 2.54 mm pitch. The supplied photograph confirms
the order. All pads are at X36.00 mm; from top to bottom:

| Pad | Net | Y, mm |
|---|---|---|
| 8 | ECG_INTB | 37.22 |
| 7 | ECG_INT2B | 39.76 |
| 6 | SPI_MISO | 42.30 |
| 5 | SPI_MOSI | 44.84 |
| 4 | SPI_SCK | 47.38 |
| 3 | ECG_CS | 49.92 |
| 2 | +3V3 / VCC | 52.46 |
| 1 | GND | 55.00 |

The daughterboard body extends left, with the ECG jack/cable toward the left
edge. Its body and cable area uses a conservative reservation. Check the first
assembled board and keep the enclosure opening adjustable.

## Manufacturing files

The `fabrication/` folder contains Gerbers, separate PTH/NPTH Excellon files,
drill map/report, schematic PDF, fabrication notes and the final DRC/ERC reports.
Gerbers include copper, solder mask, silkscreen, paste and board outline layers.

Do not panelize, alter the outline or merge the NPTH file into plated drills
without confirming the four mounting holes remain non-plated.

## First-article electrical test

1. Inspect for shorts and connector reversal before fitting modules.
2. Fit only the power input, switch and BO-10186.
3. Use a current-limited supply in place of the cell and confirm 3.3 V polarity.
4. Check regulator heating and supply droop over the intended battery range.
5. Fit the ESP32 and test BLE current bursts.
6. Fit the SD socket/card and test repeated writes while monitoring 3.3 V.
7. Fit the MAX30003 last and confirm SPI/interrupt operation before electrodes.

This is a research/evaluation prototype. It is not a certified medical device
or patient-isolated system. Disconnect electrodes during charging, USB/UART
programming or any mains-connected measurement. Use battery power while a
person is connected. Have the assembled electrode-input protection reviewed
before body-connected testing, and do not infer ECGP/ECGN/VCM from cable colours.
