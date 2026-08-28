import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from shapely.wkt import loads
from scipy.spatial import KDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# PARAMETER IDE LU (SILAKAN DIMAINKAN)
# ==========================================
INPUT_FILE = "data_tiang.csv"     # csv cluster besar (misal hasil export Code 1 yg sudah difilter 1 cluster)

BOBOT_DENSITAS = 0.2              # 0 = murni cari yang paling deket (nearest-neighbor biasa)
                                   # semakin besar (>1) = makin ngejar span yang padat tiang duluan
                                   # walau lokasinya agak lebih jauh dari cluster yg lagi kebentuk

IZINKAN_POTONG_SPAN_TERAKHIR = True  # kalau True, span terakhir boleh dipotong (ambil sebagian tiang
                                      # yang berurutan/contiguous) biar totalnya pas mendekati target

# BARU: kontrol analisa per-regional.
# MODE_REGIONAL = 'SATU'  -> cuma proses 1 regional yang kamu pilih di REGIONAL_TERPILIH,
#                            targetnya pakai TARGET_JUMLAH_TIANG (1 angka)
# MODE_REGIONAL = 'SEMUA' -> loop SEMUA regional sekaligus, dan tiap regional BOLEH beda target,
#                            atur di TARGET_PER_REGIONAL (dict). Regional yang gak disebut di situ
#                            otomatis pakai DEFAULT_TARGET_JUMLAH_TIANG.
MODE_REGIONAL = 'SEMUA'

# --- dipakai kalau MODE_REGIONAL = 'SATU' ---
REGIONAL_TERPILIH = 'R06 JAWA TENGAH'
TARGET_JUMLAH_TIANG = 60

# --- dipakai kalau MODE_REGIONAL = 'SEMUA' ---
TARGET_PER_REGIONAL = {
    'R04-JABODETABEK': 7040,
    'R05-JABAR': 12271,
    'R06-JATENG': 14790,
    'R07-JATIM': 7832,
    # 'R07 JAWA TIMUR': 100,   <- tinggal tambah baris begini per regional, angka boleh beda-beda
}
DEFAULT_TARGET_JUMLAH_TIANG = 50   # target buat regional yang gak ditulis di TARGET_PER_REGIONAL

# BARU: kesadaran geografis "pulau" (biar cluster nggak lompat nyebrang laut).
# Ini pakai pendekatan JARAK (gap), bukan data batas pulau/GIS beneran (karena kita gak punya file
# batas pulau) -> 2 span dianggap "1 pulau/daratan yang sama" kalau rantai jarak terdekatnya di bawah
# PULAU_GAP_KM. Span yang jaraknya lebih jauh dari itu ke SEMUA span lain dianggap pulau terpisah.
PULAU_GAP_KM = 15.0        # naikkan kalau jaringan kamu wajar punya span yg jauhan tapi masih 1 pulau
                            # (misal daerah pegunungan/pedesaan jarang tiang); turunkan kalau pulau-pulau
                            # di data kamu emang berdekatan (misal antar pulau kecil yg selat-nya sempit)
IZINKAN_LOMPAT_PULAU = True  # kalau tiang di pulau yg sama gak cukup buat capai target, boleh lanjut
                              # ambil dari pulau lain (True) atau berhenti apa adanya (False)?

BUAT_PETA_INTERAKTIF = True  # bikin file .html peta OpenStreetMap beneran (folium) per regional,
                              # buat ngecek visual apakah ada titik yang "nyasar" ke laut/pulau lain
                              # (butuh: !pip install folium -q, dan koneksi internet pas dibuka)

# BARU: kontrol titik awal (seed/benih) pencarian, berdasarkan kolom 'stort'.
# MODE_SEED = 'OTOMATIS'    -> pilih span paling padat SE-REGIONAL (default, kaya sebelumnya)
# MODE_SEED = 'PILIH_STORT' -> paksa mulai dari salah satu STORT tertentu (misal 'BYL1' atau 'KDS1'),
#                               lalu di dalam stort itu span paling padat yang jadi benihnya.
MODE_SEED = 'PILIH_STORT'
SEED_STORT_TERPILIH = 'BYL1'        # dipakai kalau MODE_REGIONAL='SATU' & MODE_SEED='PILIH_STORT'
SEED_STORT_PER_REGIONAL = {         # dipakai kalau MODE_REGIONAL='SEMUA' & MODE_SEED='PILIH_STORT'.
    'R04-JABODETABEK': 'KYB0',
    'R05-JABAR': 'BDG0',
    'R06-JATENG': 'SKT0',
    'R07-JATIM': 'KPN1',      # Regional yang gak disebut di sini otomatis fallback ke span paling
    # 'R07 JAWA TIMUR': 'STORT_X',  # padat se-regional (mode OTOMATIS) buat regional itu aja.
}

# BARU: RESTRICTION / PENGECUALIAN STORT.
# Stort yang didaftar di sini akan DIHIRAUKAN TOTAL: span-nya diblokir PERMANEN dari proses
# pertumbuhan cluster (walau lokasinya deket/padat banget ke cluster yang lagi kebentuk), jadi
# DIJAMIN gak akan pernah masuk kategori "Inner (Terpilih)". Tiang-tiangnya tetap ada di
# dataframe hasil & di plot/CSV (biar kelihatan), tapi statusnya selalu "Outer (Tidak Terpilih)".
# Cocok buat exclude STORT yang emang gak boleh ikut proyek/batch ini.
# Pencocokan nama stort case-insensitive & spasi di ujung diabaikan (sama kayak SEED_STORT).
STORT_DIKECUALIKAN_TERPILIH = []     # dipakai kalau MODE_REGIONAL='SATU', isi list, misal: ['BYL2', 'BYL3']
STORT_DIKECUALIKAN_PER_REGIONAL = {  # dipakai kalau MODE_REGIONAL='SEMUA'
    # 'R04-JABODETABEK': ['STORT_X', 'STORT_Y'],
    # 'R06-JATENG': ['SKT2'],
    # Regional yang gak disebut di sini = gak ada stort yang dikecualikan (proses normal semua).
    'R07-JATIM': ['BKL1'],
}
# ==========================================

print("Memuat dataset...")
try:
    df_full = pd.read_csv(INPUT_FILE)
    df_full['geometry'] = df_full['wkt'].apply(loads)
    df_full['lon'] = df_full['geometry'].apply(lambda p: p.x)
    df_full['lat'] = df_full['geometry'].apply(lambda p: p.y)
except Exception as e:
    print(f"Pastikan file '{INPUT_FILE}' sudah di-upload. Error: {e}")
    import sys; sys.exit()

if 'regional' not in df_full.columns:
    print("Kolom 'regional' tidak ditemukan, analisa dijalankan gabungan (1 grup: 'SEMUA').")
    df_full['regional'] = 'SEMUA'

daftar_regional_tersedia = sorted(df_full['regional'].dropna().unique())
print(f"Regional yang tersedia di file ini: {daftar_regional_tersedia}")

if MODE_REGIONAL == 'SATU':
    if REGIONAL_TERPILIH not in daftar_regional_tersedia:
        print(f"\n⚠️  REGIONAL_TERPILIH='{REGIONAL_TERPILIH}' tidak ada di data. "
              f"Pilih salah satu dari: {daftar_regional_tersedia}")
        import sys; sys.exit()
    daftar_regional_proses = [REGIONAL_TERPILIH]
    target_per_reg_final = {REGIONAL_TERPILIH: TARGET_JUMLAH_TIANG}
else:
    daftar_regional_proses = daftar_regional_tersedia
    # cek typo di TARGET_PER_REGIONAL: key yg gak match nama regional manapun di data
    key_ngawur = [k for k in TARGET_PER_REGIONAL if k not in daftar_regional_tersedia]
    if key_ngawur:
        print(f"⚠️  Key di TARGET_PER_REGIONAL ini gak ketemu di data (dicek typo-nya): {key_ngawur}")
    target_per_reg_final = {
        reg: TARGET_PER_REGIONAL.get(reg, DEFAULT_TARGET_JUMLAH_TIANG)
        for reg in daftar_regional_tersedia
    }

print(f"Regional yang akan diproses & targetnya:")
for reg in daftar_regional_proses:
    tag = '' if reg in TARGET_PER_REGIONAL or MODE_REGIONAL == 'SATU' else ' (pakai default)'
    print(f"   - {reg}: {target_per_reg_final[reg]} tiang{tag}")
print()

# --- tentukan titik awal (seed/stort) tiap regional ---
if MODE_SEED == 'PILIH_STORT':
    if MODE_REGIONAL == 'SATU':
        stort_final = {REGIONAL_TERPILIH: SEED_STORT_TERPILIH}
    else:
        stort_final = {reg: SEED_STORT_PER_REGIONAL.get(reg, None) for reg in daftar_regional_proses}
else:
    stort_final = {reg: None for reg in daftar_regional_proses}

print(f"Titik awal (seed) tiap regional:")
for reg in daftar_regional_proses:
    print(f"   - {reg}: {stort_final[reg] if stort_final[reg] else 'Otomatis (paling padat se-regional)'}")
print()

# --- BARU: tentukan stort yang DIKECUALIKAN (restriction) tiap regional ---
if MODE_REGIONAL == 'SATU':
    stort_dikecualikan_final = {
        REGIONAL_TERPILIH: [str(s).strip() for s in STORT_DIKECUALIKAN_TERPILIH]
    }
else:
    key_ngawur_excl = [k for k in STORT_DIKECUALIKAN_PER_REGIONAL if k not in daftar_regional_tersedia]
    if key_ngawur_excl:
        print(f"⚠️  Key di STORT_DIKECUALIKAN_PER_REGIONAL ini gak ketemu di data (dicek typo-nya): {key_ngawur_excl}")
    stort_dikecualikan_final = {
        reg: [str(s).strip() for s in STORT_DIKECUALIKAN_PER_REGIONAL.get(reg, [])]
        for reg in daftar_regional_proses
    }

print(f"STORT yang dikecualikan/dihiraukan (dijamin gak akan masuk Inner) tiap regional:")
for reg in daftar_regional_proses:
    daftar_excl = stort_dikecualikan_final[reg]
    print(f"   - {reg}: {daftar_excl if daftar_excl else '(tidak ada, semua stort diproses normal)'}")
print()

pola_nama = re.compile(r'^(?P<span>.+)-(?P<seg>\d+)-T(?P<seq>\d+)$')

def bedah_nama(nama):
    m = pola_nama.match(str(nama))
    if m:
        return int(m.group('seg')), int(m.group('seq'))
    return -1, -1

def cari_seed_span(df_r, span_ids, span_counts, stort_pilihan, reg_name):
    """Tentukan span benih: otomatis (paling padat se-regional) ATAU dipaksa mulai dari
    dalam 1 STORT tertentu (lalu di dalam stort itu span paling padat yg jadi benihnya)."""
    if not stort_pilihan:
        return int(np.argmax(span_counts)), 'OTOMATIS (paling padat se-regional)'
    if 'stort' not in df_r.columns:
        print(f"      ⚠️  Kolom 'stort' gak ada di data, SEED_STORT diabaikan -> pakai span terpadat se-regional.")
        return int(np.argmax(span_counts)), 'OTOMATIS (kolom stort tidak ada)'
    stort_tersedia = sorted(df_r['stort'].dropna().unique())
    cocok = [s for s in stort_tersedia if str(s).strip().lower() == str(stort_pilihan).strip().lower()]
    if not cocok:
        print(f"      ⚠️  SEED_STORT='{stort_pilihan}' gak ketemu di regional '{reg_name}'. "
              f"Stort yang tersedia di sini: {stort_tersedia}. Fallback ke span terpadat se-regional.")
        return int(np.argmax(span_counts)), 'OTOMATIS (stort tidak ditemukan)'
    stort_asli = cocok[0]
    span_dlm_stort = set(df_r.loc[df_r['stort'] == stort_asli, 'spanid'].unique())
    idx_kandidat = [i for i, sid in enumerate(span_ids) if sid in span_dlm_stort]
    if not idx_kandidat:
        print(f"      ⚠️  Gak ada span dengan stort='{stort_asli}'. Fallback ke span terpadat se-regional.")
        return int(np.argmax(span_counts)), 'OTOMATIS (span kosong)'
    idx_terbaik_relatif = int(np.argmax(span_counts[idx_kandidat]))
    seed_local = idx_kandidat[idx_terbaik_relatif]
    return seed_local, f"STORT='{stort_asli}' (paling padat di situ)"

# BARU: cari span-span yang HARUS diblokir permanen (gak boleh pernah masuk Inner) karena
# stort-nya ada di daftar pengecualian regional ini.
def cari_span_diblokir(df_r, span_ids, stort_dikecualikan, reg_name):
    n_span = len(span_ids)
    span_diblokir_mask = np.zeros(n_span, dtype=bool)
    if not stort_dikecualikan:
        return span_diblokir_mask, set(), 0
    if 'stort' not in df_r.columns:
        print(f"      ⚠️  Kolom 'stort' gak ada di data, STORT_DIKECUALIKAN diabaikan (gak ada span yang diblokir).")
        return span_diblokir_mask, set(), 0
    stort_tersedia = sorted(df_r['stort'].dropna().unique())
    target_excl_lower = {s.lower() for s in stort_dikecualikan}
    stort_cocok = [s for s in stort_tersedia if str(s).strip().lower() in target_excl_lower]
    ketemu_lower = {str(s).strip().lower() for s in stort_cocok}
    tidak_ketemu = sorted(target_excl_lower - ketemu_lower)
    if tidak_ketemu:
        print(f"      ⚠️  STORT_DIKECUALIKAN berikut gak ketemu di regional '{reg_name}' (dicek typo-nya): {tidak_ketemu}")
    if not stort_cocok:
        return span_diblokir_mask, set(), 0
    span_ids_diblokir = set(df_r.loc[df_r['stort'].isin(stort_cocok), 'spanid'].unique())
    span_diblokir_mask = np.isin(span_ids, list(span_ids_diblokir))
    n_tiang_diblokir = int(df_r['spanid'].isin(span_ids_diblokir).sum())
    print(f"      🚫 STORT dikecualikan: {stort_cocok} -> {int(span_diblokir_mask.sum())} span diblokir "
          f"({n_tiang_diblokir} tiang) -> dijamin gak akan masuk Inner.")
    return span_diblokir_mask, span_ids_diblokir, n_tiang_diblokir

def proses_satu_regional(df_r, TARGET_JUMLAH_TIANG, BOBOT_DENSITAS, IZINKAN_POTONG_SPAN_TERAKHIR,
                          PULAU_GAP_KM, IZINKAN_LOMPAT_PULAU, stort_pilihan, reg_name,
                          stort_dikecualikan=None):
    """Reverse-cluster growth di dalam 1 subset regional saja.
    CATATAN PERFORMA: pertumbuhan cluster dilakukan di LEVEL SPAN (bukan level tiang per tiang),
    pakai pendekatan mirip algoritma Prim (MST) yang divektorisasi numpy. Skor tiap span kandidat
    cuma dibandingkan ke centroid span yang BARU DITAMBAHKAN (bukan hitung ulang ke SEMUA tiang
    terpilih tiap iterasi) -> dari O(spans_tersisa x tiang_terpilih) per iterasi jadi O(jumlah_span)
    per iterasi. Ini yang bikin kuat sampai puluhan ribu tiang / ribuan span.
    CATATAN PULAU: sebelum tumbuh, span-span dikelompokkan jadi "pulau" (macro-cluster) pakai
    connectivity radius PULAU_GAP_KM (logikanya sama kaya Code 1, tapi di level span). Pertumbuhan
    FASE 1 dibatasi cuma ke span yang 1 pulau sama span benih. Kalau tiangnya kurang dari target,
    baru (kalau diizinkan) FASE 2 lanjut ambil dari pulau lain.
    CATATAN RESTRICTION: kalau stort_dikecualikan diisi, span yang stort-nya masuk daftar itu
    diberi flag "diblokir" dan DIKELUARKAN dari kandidat di setiap iterasi pertumbuhan (baik
    FASE 1 maupun FASE 2 lompat pulau) -> tiang-tiangnya tetap ada di dataframe hasil (biar
    kelihatan di plot/CSV), tapi selamanya berstatus 'Outer (Tidak Terpilih)'.
    """
    df_r = df_r.reset_index(drop=True)
    n = len(df_r)
    coords = df_r[['lon', 'lat']].values
    seg_seq = df_r['name'].apply(bedah_nama)
    df_r['kode_segmen'] = seg_seq.apply(lambda t: t[0])
    df_r['urutan_tiang'] = seg_seq.apply(lambda t: t[1])
    df_r['urutan_fisik'] = df_r['kode_segmen'] * 10000 + df_r['urutan_tiang']

    span_count_s = df_r.groupby('spanid').size()
    span_centroid_df = df_r.groupby('spanid')[['lon', 'lat']].mean()
    span_ids = span_centroid_df.index.to_numpy()
    span_coords = span_centroid_df[['lon', 'lat']].to_numpy()
    span_counts = span_count_s.loc[span_ids].to_numpy()
    n_span = len(span_ids)

    seed_local, keterangan_seed = cari_seed_span(df_r, span_ids, span_counts, stort_pilihan, reg_name)
    seed_span = span_ids[seed_local]
    print(f"   🌱 Span benih [{keterangan_seed}]: {seed_span} ({span_counts[seed_local]} tiang)")

    # --- BARU: hitung span yang diblokir total (restriction stort) ---
    span_diblokir_mask, span_ids_diblokir, n_tiang_diblokir = cari_span_diblokir(
        df_r, span_ids, stort_dikecualikan, reg_name
    )
    if span_diblokir_mask[seed_local]:
        print(f"      ⚠️  PERHATIAN: span benih {seed_span} kebetulan termasuk STORT yang dikecualikan! "
              f"Seed tetap dipakai (sesuai SEED_STORT pilihanmu), tapi span LAIN dari stort "
              f"yang dikecualikan tetap diblokir dan gak akan ditambahkan.")

    # --- deteksi "pulau" (macro-cluster) di level span ---
    span_kdtree = KDTree(span_coords)
    pulau_radius_deg = PULAU_GAP_KM / 111.0
    pairs = span_kdtree.query_pairs(r=pulau_radius_deg, output_type='ndarray')
    if len(pairs) > 0:
        row = np.concatenate([pairs[:, 0], pairs[:, 1]])
        col = np.concatenate([pairs[:, 1], pairs[:, 0]])
        graph = coo_matrix((np.ones(len(row)), (row, col)), shape=(n_span, n_span))
    else:
        graph = coo_matrix((n_span, n_span))
    n_pulau, pulau_id = connected_components(csgraph=graph, directed=False)
    pulau_seed = pulau_id[seed_local]
    ukuran_pulau_seed = int(span_counts[pulau_id == pulau_seed].sum())
    print(f"   🏝️  Terdeteksi {n_pulau} kelompok pulau (gap > {PULAU_GAP_KM} km dianggap pisah). "
          f"Pulau tempat span benih: {ukuran_pulau_seed} tiang tersedia.")

    EPS_KM = 0.01
    selected_mask = np.zeros(n_span, dtype=bool)
    selected_mask[seed_local] = True
    # skor_jarak[s] = jarak (km) dari centroid span s ke centroid span TERPILIH TERDEKAT sejauh ini
    skor_jarak = np.sqrt(((span_coords - span_coords[seed_local]) ** 2).sum(axis=1)) * 111.0
    span_terpilih_local = [seed_local]
    total_terpilih = int(span_counts[seed_local])
    last_added_local = seed_local
    lompat_pulau_terjadi = False

    def tumbuh(mask_boleh_dipilih, label_fase):
        nonlocal total_terpilih, last_added_local
        while total_terpilih < TARGET_JUMLAH_TIANG:
            # BARU: span yang diblokir (restriction stort) selalu dikeluarkan dari kandidat,
            # di fase manapun (sama pulau ATAU lompat pulau).
            kandidat = mask_boleh_dipilih & (~selected_mask) & (~span_diblokir_mask)
            if not kandidat.any():
                return False  # kehabisan kandidat di fase ini
            d_baru = np.sqrt(((span_coords - span_coords[last_added_local]) ** 2).sum(axis=1)) * 111.0
            skor_jarak[:] = np.minimum(skor_jarak, d_baru)
            prioritas = (span_counts.astype(float) ** BOBOT_DENSITAS) / (skor_jarak + EPS_KM)
            prioritas_masked = np.where(kandidat, prioritas, -np.inf)
            span_terbaik_local = int(np.argmax(prioritas_masked))
            selected_mask[span_terbaik_local] = True
            span_terpilih_local.append(span_terbaik_local)
            total_terpilih += int(span_counts[span_terbaik_local])
            last_added_local = span_terbaik_local
            tanda = '🏝️ ' if label_fase == 'LOMPAT' else '   '
            print(f"      {tanda}+ Tambah span {span_ids[span_terbaik_local]} "
                  f"({span_counts[span_terbaik_local]} tiang, jarak {skor_jarak[span_terbaik_local]:.2f} km) "
                  f"-> total {total_terpilih} tiang")
        return True

    # FASE 1: tumbuh HANYA di pulau yang sama dengan span benih
    tercapai = tumbuh(pulau_id == pulau_seed, 'SAMA')

    # FASE 2 (opsional): kalau belum cukup & masih ada span (yang gak diblokir) di pulau lain
    if not tercapai and total_terpilih < TARGET_JUMLAH_TIANG:
        if IZINKAN_LOMPAT_PULAU and (~selected_mask & ~span_diblokir_mask).any():
            print(f"      ⚠️  Tiang di pulau yang sama cuma {total_terpilih}, kurang dari target "
                  f"{TARGET_JUMLAH_TIANG}. Lanjut ambil dari pulau lain...")
            lompat_pulau_terjadi = True
            tumbuh(np.ones(n_span, dtype=bool), 'LOMPAT')
        else:
            print(f"      ⚠️  Tiang di pulau yang sama cuma {total_terpilih} dari target {TARGET_JUMLAH_TIANG} "
                  f"tiang, dan (IZINKAN_LOMPAT_PULAU=False atau semua span sisanya diblokir) -> berhenti di sini.")

    span_terpilih = [span_ids[i] for i in span_terpilih_local]
    span_terpilih_pulau_sama = set(span_ids[i] for i in span_terpilih_local if pulau_id[i] == pulau_seed)
    tiang_terpilih_idx = set(df_r.index[df_r['spanid'].isin(span_terpilih)])
    total_terpilih = len(tiang_terpilih_idx)

    kelebihan = total_terpilih - TARGET_JUMLAH_TIANG
    if IZINKAN_POTONG_SPAN_TERAKHIR and kelebihan > 0:
        span_dipotong = span_terpilih[-1]
        idx_span_dipotong = df_r.index[df_r['spanid'] == span_dipotong].tolist()
        idx_span_lain = tiang_terpilih_idx - set(idx_span_dipotong)
        coords_span = coords[idx_span_dipotong]
        if idx_span_lain:
            acuan = coords[list(idx_span_lain)]
        else:
            acuan = coords_span.mean(axis=0, keepdims=True)
        dmin_per_tiang = np.array([
            np.sqrt(((acuan - c) ** 2).sum(axis=1)).min() for c in coords_span
        ])
        entry_local = np.argmin(dmin_per_tiang)
        entry_idx = idx_span_dipotong[entry_local]
        entry_urutan = df_r.loc[entry_idx, 'urutan_fisik']
        perlu = max(TARGET_JUMLAH_TIANG - len(idx_span_lain), 0)
        df_span_pot = df_r.loc[idx_span_dipotong].copy()
        df_span_pot['selisih_urutan'] = (df_span_pot['urutan_fisik'] - entry_urutan).abs()
        idx_dipertahankan = df_span_pot.sort_values('selisih_urutan').head(perlu).index
        tiang_terpilih_idx = idx_span_lain | set(idx_dipertahankan)
        total_terpilih = len(tiang_terpilih_idx)
        print(f"      ✂️  Span {span_dipotong} dipotong: dipertahankan {len(idx_dipertahankan)} dari "
              f"{len(idx_span_dipotong)} tiang (paling bersebelahan dgn cluster) -> total akhir {total_terpilih}")

    df_r['Kategori_Reverse'] = np.where(df_r.index.isin(tiang_terpilih_idx), 'Inner (Terpilih)', 'Outer (Tidak Terpilih)')

    # tandai tiap tiang terpilih: dari pulau yang sama dgn benih, atau hasil lompat ke pulau lain
    def tandai_pulau(row):
        if row.name not in tiang_terpilih_idx:
            return '-'
        return 'Pulau Sama (Benih)' if row['spanid'] in span_terpilih_pulau_sama else '⚠️ Lompat Pulau (Fallback)'
    df_r['Kategori_Pulau'] = df_r.apply(tandai_pulau, axis=1)

    # BARU: tandai tiang mana yang kena restriction (biar gampang diaudit di CSV export)
    if span_ids_diblokir:
        df_r['Stort_Dikecualikan'] = df_r['spanid'].isin(span_ids_diblokir)
    else:
        df_r['Stort_Dikecualikan'] = False

    n_lompat = int((df_r['Kategori_Pulau'] == '⚠️ Lompat Pulau (Fallback)').sum())
    return df_r, seed_span, span_terpilih, total_terpilih, n_pulau, lompat_pulau_terjadi, n_lompat, n_tiang_diblokir

hasil_per_regional = []
info_seed = {}
info_target = {}
for reg in daftar_regional_proses:
    target_reg = target_per_reg_final[reg]
    df_r = df_full[df_full['regional'] == reg].copy()
    print(f"📍 Regional: {reg}  (total {len(df_r)} tiang tersedia, target {target_reg} tiang)")
    df_r, seed_span, span_terpilih, total_terpilih, n_pulau, lompat_terjadi, n_lompat, n_diblokir = proses_satu_regional(
        df_r, target_reg, BOBOT_DENSITAS, IZINKAN_POTONG_SPAN_TERAKHIR, PULAU_GAP_KM, IZINKAN_LOMPAT_PULAU,
        stort_final[reg], reg, stort_dikecualikan_final[reg]
    )
    ringkas_pulau = f", ⚠️ {n_lompat} tiang dari pulau lain" if lompat_terjadi else ", semua dari 1 pulau yang sama ✅"
    ringkas_restriksi = f", 🚫 {n_diblokir} tiang diabaikan (restriction stort)" if n_diblokir > 0 else ""
    print(f"   ✅ Target {target_reg} tiang -> terkumpul {total_terpilih} tiang dari {len(span_terpilih)} "
          f"span ({n_pulau} pulau terdeteksi di regional ini{ringkas_pulau}{ringkas_restriksi})\n")
    info_seed[reg] = seed_span
    info_target[reg] = target_reg
    hasil_per_regional.append(df_r)

df_all = pd.concat(hasil_per_regional, ignore_index=True)

# ==========================================
# VISUALISASI: 1 PANEL PER REGIONAL YANG DIPROSES
# ==========================================
n_reg = len(daftar_regional_proses)
n_cols = 2 if n_reg > 1 else 1
n_rows = int(np.ceil(n_reg / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 7 * n_rows), squeeze=False)
for i, reg in enumerate(daftar_regional_proses):
    ax = axes[i // n_cols][i % n_cols]
    df_r = df_all[df_all['regional'] == reg]
    outer = df_r[df_r['Kategori_Reverse'] == 'Outer (Tidak Terpilih)']
    inner_sama = df_r[df_r['Kategori_Pulau'] == 'Pulau Sama (Benih)']
    inner_lompat = df_r[df_r['Kategori_Pulau'] == '⚠️ Lompat Pulau (Fallback)']
    ax.scatter(outer['lon'], outer['lat'], c='#bdc3c7', s=15, alpha=0.6, label=f'Outer ({len(outer)})', zorder=1)
    ax.scatter(inner_sama['lon'], inner_sama['lat'], c='#3498db', s=25, edgecolors='black', linewidths=0.4,
               label=f'Inner - Pulau Sama ({len(inner_sama)})', zorder=2)
    if len(inner_lompat) > 0:
        ax.scatter(inner_lompat['lon'], inner_lompat['lat'], c='#e67e22', s=35, marker='D', edgecolors='black',
                   linewidths=0.5, label=f'⚠️ Inner - Lompat Pulau ({len(inner_lompat)})', zorder=3)
    seed_pts = df_r[df_r['spanid'] == info_seed[reg]]
    ax.scatter(seed_pts['lon'], seed_pts['lat'], c='#f1c40f', s=60, marker='*',
               edgecolors='black', linewidths=0.5, label=f'Span Benih: {info_seed[reg]}', zorder=4)
    ax.set_title(f'{reg}\nTarget: {info_target[reg]} tiang (terkumpul {len(inner_sama)+len(inner_lompat)})',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend(loc='upper right', framealpha=0.9, fontsize=8)
    ax.grid(True, linestyle=':', alpha=0.7)
for j in range(n_reg, n_rows * n_cols):
    axes[j // n_cols][j % n_cols].axis('off')
plt.tight_layout()
plt.show()

# ==========================================
# EXPORT CSV
# ==========================================
df_export = df_all.drop(columns=['geometry', 'lon', 'lat'])
if MODE_REGIONAL == 'SEMUA':
    tag_regional = 'semua_regional_target_variatif'
else:
    tag_regional = f"{REGIONAL_TERPILIH.replace(' ', '_')}_target{TARGET_JUMLAH_TIANG}"
export_filename = f"TiangJabo_Reverse_{tag_regional}.csv"
df_export.to_csv(export_filename, index=False)
print(f"✅ File CSV berhasil disimpan: {export_filename}")

# ==========================================
# PETA INTERAKTIF (folium) - buat ngecek visual di atas basemap OpenStreetMap asli,
# jadi kalau ada tiang yang "nyasar" ke laut/pulau lain bakal kelihatan jelas.
# ==========================================
if BUAT_PETA_INTERAKTIF:
    try:
        import folium
        MAKS_TITIK_SAMA_PULAU = 3000  # batas jumlah titik "Pulau Sama" yg digambar 1-1 biar peta tetap ringan
                                       # (titik ⚠️ Lompat Pulau selalu digambar 100%, apapun jumlahnya)
        for reg in daftar_regional_proses:
            df_r = df_all[df_all['regional'] == reg]
            inner_sama = df_r[df_r['Kategori_Pulau'] == 'Pulau Sama (Benih)']
            inner_lompat = df_r[df_r['Kategori_Pulau'] == '⚠️ Lompat Pulau (Fallback)']
            seed_pts = df_r[df_r['spanid'] == info_seed[reg]]
            if len(inner_sama) > MAKS_TITIK_SAMA_PULAU:
                inner_sama_peta = inner_sama.sample(MAKS_TITIK_SAMA_PULAU, random_state=42)
                catatan_sampling = f" (ditampilkan sampel {MAKS_TITIK_SAMA_PULAU} dari {len(inner_sama)} biar peta ringan)"
            else:
                inner_sama_peta = inner_sama
                catatan_sampling = ""
            pusat_lat = df_r['lat'].mean()
            pusat_lon = df_r['lon'].mean()
            m = folium.Map(location=[pusat_lat, pusat_lon], zoom_start=9, tiles='OpenStreetMap')
            fg_sama = folium.FeatureGroup(name=f'Inner - Pulau Sama ({len(inner_sama)}){catatan_sampling}')
            for _, row in inner_sama_peta.iterrows():
                folium.CircleMarker(
                    location=[row['lat'], row['lon']], radius=3, color='#3498db',
                    fill=True, fill_color='#3498db', fill_opacity=0.8, weight=1,
                    popup=str(row['name']),
                ).add_to(fg_sama)
            fg_sama.add_to(m)
            if len(inner_lompat) > 0:
                fg_lompat = folium.FeatureGroup(name=f'⚠️ Inner - Lompat Pulau ({len(inner_lompat)})')
                for _, row in inner_lompat.iterrows():
                    folium.CircleMarker(
                        location=[row['lat'], row['lon']], radius=6, color='#e67e22',
                        fill=True, fill_color='#e67e22', fill_opacity=0.95, weight=2,
                        popup=f"⚠️ LOMPAT PULAU: {row['name']}",
                    ).add_to(fg_lompat)
                fg_lompat.add_to(m)
            for _, row in seed_pts.iterrows():
                folium.Marker(
                    location=[row['lat'], row['lon']],
                    icon=folium.Icon(color='orange', icon='star'),
                    popup=f"Span Benih: {info_seed[reg]}",
                ).add_to(m)
            folium.LayerControl(collapsed=False).add_to(m)
            nama_file_peta = f"Peta_Reverse_{reg.replace(' ', '_')}_{tag_regional}.html"
            m.save(nama_file_peta)
            print(f"🗺️  Peta interaktif disimpan: {nama_file_peta} (buka di browser buat ngecek visual)")
    except ImportError:
        print("\n⚠️  Package 'folium' belum ke-install, peta interaktif dilewati.")
        print("    Jalankan `!pip install folium -q` di cell terpisah lalu run ulang script ini kalau mau peta-nya.")
