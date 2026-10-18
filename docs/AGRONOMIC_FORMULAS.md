# Agronomic Formulas & Mathematical Models

This document details the agricultural physics and meteorological equations implemented across the Smart Irrigation platform.

---

## 1. Reference Evapotranspiration ($ET_0$)

### FAO-56 Penman-Monteith Formulation
The standardized FAO-56 Penman-Monteith equation estimates the evaporative demand of a reference grass crop:

$$ET_0 = \frac{0.408 \Delta (R_n - G) + \gamma \frac{900}{T + 273} u_2 (e_s - e_a)}{\Delta + \gamma (1 + 0.34 u_2)}$$

Where:
- $ET_0$: Reference evapotranspiration [$\text{mm day}^{-1}$]
- $R_n$: Net radiation at the crop surface [$\text{MJ m}^{-2}\text{ day}^{-1}$]
- $G$: Soil heat flux density [$\text{MJ m}^{-2}\text{ day}^{-1}$] ($\approx 0$ for daily calculations)
- $T$: Mean daily air temperature at 2 m height [°C]
- $u_2$: Wind speed at 2 m height [$\text{m s}^{-1}$]
- $e_s$: Saturation vapour pressure [$\text{kPa}$], calculated via Tetens formula:
  $$e_s = 0.6108 \exp\left(\frac{17.27 T}{T + 237.3}\right)$$
- $e_a$: Actual vapour pressure [$\text{kPa}$], derived from relative humidity ($RH$):
  $$e_a = e_s \times \frac{RH}{100}$$
- $\Delta$: Slope of saturation vapour pressure curve [$\text{kPa }^\circ\text{C}^{-1}$]:
  $$\Delta = \frac{4098 e_s}{(T + 237.3)^2}$$
- $\gamma$: Psychrometric constant [$\text{kPa }^\circ\text{C}^{-1}$] ($\approx 0.067\text{ kPa }^\circ\text{C}^{-1}$ at sea level).

---

## 2. Crop Water Requirement ($ET_c$)

$$ET_c = K_c \times ET_0$$

Where $K_c$ is the crop coefficient varying by phenological growth stage:
1. **Initial stage**: Emergence and early ground cover (typical $K_c \in [0.35, 0.45]$)
2. **Crop development stage**: Rapid vegetative growth to full cover ($K_c \in [0.70, 0.85]$)
3. **Mid-season stage**: Flowering, fruiting, and grain fill (peak demand $K_c \in [1.05, 1.25]$)
4. **Late season stage**: Maturation and senescence ($K_c \in [0.40, 0.75]$)

---

## 3. Effective Rainfall ($P_e$)

### USDA Soil Conservation Service (SCS) Empirical Method
Not all precipitation infiltrates into the root zone; high intensity storms trigger runoff, while light drizzles evaporate before reaching roots.

$$\begin{cases}
P_e = P_{forecast} \times \frac{125 - 0.2 \times P_{forecast}}{125} & \text{for } P_{forecast} \le 75\text{ mm} \\
P_e = \frac{125}{3} + 0.1 \times P_{forecast} & \text{for } P_{forecast} > 75\text{ mm}
\end{cases}$$

---

## 4. Soil Water Contribution & Available Water Capacity (AWC)

$$\text{AWC} = \text{Field Capacity (FC)} - \text{Permanent Wilting Point (PWP)}$$

Available water storage across soil textures:
- **Sandy**: $\approx 60\text{ mm/m}$
- **Sandy Loam**: $\approx 100\text{ mm/m}$
- **Loamy**: $\approx 140\text{ mm/m}$
- **Clay Loam / Silty**: $\approx 160\text{ mm/m}$
- **Clay**: $\approx 180\text{ mm/m}$

Usable soil water reserve in active root zone:
$$\text{Usable Soil Water (mm)} = \max\left(0, (\theta_{current} - \theta_{pwp}) \times 0.1 \times Z_r\right)$$
Where $Z_r$ is effective root depth in cm.

---

## 5. Net and Gross Irrigation Requirements

### Net Irrigation ($I_{net}$):
$$I_{net} = \max\left(0, ET_c - P_e - \text{Usable Soil Water}\right)$$

### Gross Irrigation ($I_{gross}$):
Accounts for application uniformity and conveyance losses:
$$I_{gross} = \frac{I_{net}}{\eta_{irr}}$$

Typical Irrigation Efficiencies ($\eta_{irr}$):
- **Drip / Micro-irrigation**: $90\%$ ($\eta = 0.90$)
- **Sprinkler**: $75\%$ ($\eta = 0.75$)
- **Furrow**: $60\%$ ($\eta = 0.60$)
- **Basin / Flood**: $50\%$ ($\eta = 0.50$)

---

## 6. Depth to Volume Conversions

$$1\text{ mm depth over } 1\text{ m}^2 = 1\text{ Liter}$$

Area equivalents:
- $1\text{ Hectare} = 10,000\text{ m}^2$
- $1\text{ Acre} \approx 4,046.86\text{ m}^2$

Total Water Volume:
$$\text{Volume (Liters)} = \text{Depth (mm)} \times \text{Area (m}^2\text{)}$$
$$\text{Volume (m}^3\text{)} = \frac{\text{Volume (Liters)}}{1000}$$
