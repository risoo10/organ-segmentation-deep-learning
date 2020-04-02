# Konzultacie 25.3.2020

## Aktualny stav

* [x] 3D konektivita - Implementacia `3d-connected-components` (dve fazy, Rosenfeld and Pflatz augmented with Union-Find)
    - DICE: 0.96736 > 0.96769   
    - Zlepsenie ~ 0.0003
* [x] Zaplnenie dier - Visualization Toolkit VTK  - Vytvorenie povrchu (ISOSURFACE)
* [x] Synchronizovana Vizualizacia 3D - VTK
* [x] Preprocessing a registracia

    - Rotacia (nekonzistentne)
    - Normalizacia
    - Clamping <-150, 400>
    - Rozdelenie:


| Mnozina     | Pomer    | Pocet pacientov |
|-------------|----------|-----------------|
| TRAIN       | 70%      | 92              |
| VALID       | 10%      | 13              |
| TEST        | 20%      | 26              |
| **Celkovo** | **100%** | **131**         |



* [x] Pred trenovanim vyrez ROI



## Plan
* [ ] Trenovanie UNET WeightDICE na LiTS datasete
* [ ] Sagitalny a Frontalny klasifikator UNET WeightDICE
    - Doplnenenie voxelu mocninou 2ky

