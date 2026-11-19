# Hardware bill of materials

The design budget was ₹6,000 excluding the Raspberry Pi. This is a target, not verified final expenditure. See [PCB BOM](hardware/pcb/fabrication/docs/BOM.csv).

| Component | Project role | Accounting status |
| --- | --- | --- |
| MAX30003 ECG module | Single-lead front end, SPI | Used; receipt/unit price to reconcile |
| ESP32-WROOM-32 | Acquisition and Wi-Fi transport | Used on custom carrier |
| 7Semi TPS631000 / BO-10186 | 3.3 V buck-boost supply | Used; confirm module against PCB |
| LiPo and TP4056 protected charger | Portable power | Used; runtime needs measurement |
| RA/LA/RL electrodes and leads | ECG connection | Consumable cost to reconcile |
| microSD, socket, switch, buttons, passives, headers | Replay and carrier support | Component details in PCB BOM |
| Rev B PCB and enclosure | Integration | Reported USD 90.60 order for five boards plus charges; batch cost differs from unit cost |
| Raspberry Pi 4B, 2 GB | Preprocessing/inference/app bridge | Excluded from budget |

The old AD8232/ADS1115/OLED list has been replaced because it did not describe the built MAX30003 design. No final cost total is claimed without invoices and shipping/tax allocation.
