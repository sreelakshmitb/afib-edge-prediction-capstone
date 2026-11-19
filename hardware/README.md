# Hardware source

`pcb/` contains the inspected nested Rev B fabrication-candidate archive: KiCad PCB/schematic/project, custom libraries, Gerbers/drills, BOM, connector maps, ERC/DRC reports, mechanical STEP/DXF and verification files. `archives/` preserves its original ZIP for provenance. Rev B means board revision B.

The source archive is a fabrication candidate and is not a new manufacturing approval. Compare it with the actual manufactured revision and assembly notes before ordering. Source contains provisional daughterboard envelopes and original evidence notes.

The built chain uses three RA/LA/RL electrodes, MAX30003 SPI, ESP32-WROOM-32, 7Semi BO-10186 3.3 V supply, LiPo/TP4056 and Raspberry Pi 4B. The demonstrated replay path and supplied live code use Wi-Fi; earlier BLE plans are historical.

[Live firmware](../firmware/afib_live_wifi/afib_live_wifi.ino) uses SCK 18, MISO 19, MOSI 23, ECG CS 26 and SD CS 13. Configure hotspot name/password and Pi IP before flashing. The deployed Wi-Fi SD replay sketch was not found in recovered files and must be recovered from the working project.

[BOM](../BOM.md) records current components and unresolved cost accounting. Prior MAX30003 calibration, SD read/write and replay checks are recorded bench evidence, not a human-use electrical safety certification.
