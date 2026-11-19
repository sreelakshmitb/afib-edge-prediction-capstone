# Evidence used for Rev B prototype candidate

- User-supplied BO-10186 listing screenshots: `references/image_5_listing_dimensions.png`
  and `references/image_6_listing_pitch.png`.
  They state 20 × 15 × 5 mm and breadboard-compatible 2.54 mm pitch.
- 7Semi product page:
  https://7semi.com/7semi-dc-dc-buck-boost-converter-1-6v-5-5v-in-3-3v-1-5a-out-tps631000/
- TPS631000 IC page: https://www.ti.com/product/TPS631000
- Supplied MAX30003 v3 06/26 photograph, stored in the project references.
- ProtoCentral listing:
  https://protocentral.com/product/protocentral-max30003-single-lead-ecg-breakout-board-v2/

The 10358 STEP/DXF files were not used for BO-10186 geometry because the STEP
identifies a different AP64350 converter and has a different envelope.

Verification used KiCad 10.0.6 ERC, DRC with refilled zones, schematic parity,
and an independent schematic-to-PCB pin/net comparison. See `verification/`.
