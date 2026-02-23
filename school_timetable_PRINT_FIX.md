# School Timetable – Print Right Border Fix (Professional)

**Option A (recommended):** Workspace mein fixed file ready hai:
- **`school_timetable_fixed.html`** — is file ko browser mein open karke Print Preview (Ctrl+P) check karein. Agar theek lage to isi ko `d:\school_timetable (7).html` par copy/rename kar sakte hain.

**Option B:** Apni file `d:\school_timetable (7).html` mein niche diye gaye changes **khud apply karein**.  
Pehle file ko **Copy** bana lein, phir **Find** ko **Replace** se badal dein.

---

## 1. Page / body overflow fix

**Find:**
```css
            html,
            body {
                width: 287mm;
                height: 200mm;
                margin: 0 !important;
                padding: 0 !important;
                background: white !important;
                overflow: hidden !important;
            }
```

**Replace with:**
```css
            html,
            body {
                width: 100% !important;
                max-width: 297mm !important;
                margin: 0 !important;
                padding: 0 !important;
                background: white !important;
                overflow: visible !important;
            }
```

---

## 2. Main content full width (print)

**Find:**
```css
            .main-content {
                margin-left: 0 !important;
                padding: 0 !important;
                width: 90% !important;
```
(agar `width: 100%` already hai to ye step skip karein.)

**Replace with:**
```css
            .main-content {
                margin-left: 0 !important;
                padding: 0 !important;
                width: 100% !important;
```

---

## 3. Timetable container – overflow hidden, border clear

**Find:**
```css
            .timetable-container {
                -webkit-box-shadow: none !important;
                box-shadow: none !important;
                border-radius: 0 !important;
                overflow: visible !important;
                border: 2pt solid #2c3e50 !important;
                border-right: 2pt solid #2c3e50 !important;
                width: 100% !important;
                max-width: 287mm !important;
                box-sizing: border-box !important;
            }
```

**Replace with:**
```css
            .timetable-container {
                -webkit-box-shadow: none !important;
                box-shadow: none !important;
                border-radius: 0 !important;
                overflow: hidden !important;
                border: 2pt solid #2c3e50 !important;
                width: 100% !important;
                max-width: 287mm !important;
                box-sizing: border-box !important;
            }
```

---

## 4. Table width 100% (sab views) – ye right border cut hone ka main fix hai

**Find:**
```css
            .view-classes .timetable {
                width: 287mm !important;
```

**Replace with:**
```css
            .view-classes .timetable {
                width: 100% !important;
```

**Find:**
```css
            .view-teachers .timetable {
                width: 287mm !important;
```

**Replace with:**
```css
            .view-teachers .timetable {
                width: 100% !important;
```

**Find:**
```css
            .view-individual .timetable {
                width: 287mm !important;
```

**Replace with:**
```css
            .view-individual .timetable {
                width: 100% !important;
```

---

## 5. Last column par strong right border (agar pehle se nahi hai)

**Find:** (ye block `/* ── ALL CELLS` ke turant baad hona chahiye)
```css
            /* Right edge: last column border so right side prints clearly */
            .timetable th:last-child,
            .timetable td:last-child {
                border-right: 2pt solid #2c3e50 !important;
            }
```

Agar ye block nahi hai to **`/* ── ALL CELLS: strong border on every cell ── */`** wale block ke **baad** ye add karein:

```css
            /* Right edge: last column border so right side prints clearly */
            .timetable th:last-child,
            .timetable td:last-child {
                border-right: 2pt solid #2c3e50 !important;
            }
            .timetable {
                width: 100% !important;
            }
```

(Agar `.timetable { width: 100% }` pehle se kahin hai to sirf last-child wala block add karein.)

---

## Kya hoga

- Table ab container ke andar **100%** rahega, isliye right side overflow nahi karega aur **right border sahi print** hoga.
- **Classes**, **Teachers**, aur **Individual Class** teeno views ke liye same fix lagega.
- Border **professional** dikhega: charon taraf clear aur cut-off nahi.

Save karke **Print Preview** (Ctrl+P) se check karein.
