import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.wkt import loads
from scipy.spatial import KDTree
from pyproj import Transformer
from sklearn.neighbors import NearestNeighbors
import hdbscan
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# PARAMETER IDE LU (SILAKAN DIMAINKAN)
# ==========================================
# GANTI PENDEKATAN: dulu pakai connected-components radius KAKU + koreksi manual (isolasi/rescue).
# Sekarang pakai HDBSCAN -- keluarga algoritma DENSITY-BASED yang PUNYA KONSEP NOISE secara native.
# "Outer" yang dimaksud memang BUKAN cluster -- itu titik yang gagal masuk cluster manapun (label -1).
# K-Means gak dipake krn perlu k di depan & tiap titik PASTI kebagian cluster (gak ada noise). DBSCAN
# polos juga kurang krn eps-nya GLOBAL -- padahal kepadatan tiang beda 2-3 orde besaran antar wilayah
# (Jakarta vs pelosok). HDBSCAN jalanin DBSCAN di SEMUA nilai eps sekaligus & ambil cluster paling
# stabil, jadi cluster padat di kota & cluster renggang di pinggiran bisa hidup berdampingan.
#
# Butuh: !pip install hdbscan scikit-learn pyproj -q  (jalanin di cell terpisah sebelum run script ini)
INPUT_FILE = "data_tiang.csv"     # nama file csv tiang yang sudah di-upload

MIN_CLUSTER_SIZE = 10             # brp tiang minimal supaya layak disebut "titik keramaian" -- ini
                                   # KEPUTUSAN BISNIS, bukan statistik murni. Naikkan buat lebih strict.
MIN_SAMPLES = 1                   # makin gede -> makin galak nandain noise. Sebaran tiang itu MEMANJANG
                                   # ngikutin jalan (bukan blobby kayak data pada umumnya), makanya
                                   # dipakai SEKECIL MUNGKIN (1-2) -- min_samples gede (10+) bikin
                                   # ujung-ujung baris tiang yg padat sekalipun kecap noise semua krn
                                   # tetangga di arah TEGAK LURUS jalan emang dikit.
CLUSTER_SELECTION_EPSILON_M = 800.0  # gabungin cluster yg jaraknya < ini (dalam METER) -- cegah 1
                                   # kelurahan kepecah jadi belasan cluster kecil gara-gara dikit putus
                                   # di persimpangan/gang. Divalidasi: makin gede angka ini + MIN_SAMPLES
                                   # kecil, DBCV (validitas cluster) justru NAIK -- bukan cuma "keliatan
                                   # lebih bersih" doang.
CLUSTER_SELECTION_METHOD = 'eom'  # 'eom' -> cluster gede & stabil (dipakai di sini, analisa regional
                                   # skala besar). 'leaf' -> lebih granular/detail per area kecil.

# PENTING (ketauan pas ngecek manual): titik yg keliatan "di dalam" area padat di plot skala regional
# BELUM TENTU beneran deket -- di plot yg meliputi ratusan km, gap 1-3km cuma keliatan seuprit padahal
# itu jarak asli antar jalan/kompleks yg beda. Sebelum ubah MIN_SAMPLES/EPSILON lebih jauh, ZOOM IN dulu
# ke titik yg dicurigai & cek jarak aslinya (lihat percakapan/analisa) -- kalau makin dilonggarin gak
# ngefek, itu tandanya emang jauh beneran, bukan soal parameter.

# Definisi "outer" disusun BERLAPIS (biar gampang dijelasin ke stakeholder), bukan 1 kriteria doang:
#
#  Layer 1 KERAS   -> label HDBSCAN == -1. Titik yg beneran gak nyambung ke kepadatan manapun.
#  Layer 2 LUNAK   -> skor GLOSH (outlier_scores_) di atas persentil ini, DI ANTARA titik yg sudah
#                     masuk cluster (bukan noise). Nangkep titik yg secara teknis masuk cluster tapi
#                     ada di pinggirannya. Set None buat matiin layer ini.
#  Layer 3 ATURAN BISNIS -> jarak cluster ke cluster VALID lain manapun melebihi JARAK_ATURAN_BISNIS_KM,
#                     KECUALI ukuran cluster itu sendiri udah >= BATAS_MANDIRI_TERISOLASI (dianggap
#                     emang kota/jaringan lain yg sah, walau lokasinya jauh sendirian). Ini nangkep
#                     kasus yang HDBSCAN gak bisa nangkep sendiri: cluster yang PADAT SECARA LOKAL
#                     (makanya gak kena noise) tapi kepencil jauh dari jaringan padat manapun -- paling
#                     gampang dipertanggungjawabkan ke tim lapangan krn satuannya konkret (km).
#  Layer 4 RESCUE - DIAPIT TETANGGA -> BARU. Titik NOISE (Layer 1) yang DIKELILINGI (diapit) minimal
#                     MIN_TETANGGA_RESCUE tiang non-noise dalam radius JARAK_RESCUE_DEKAT_KM diselamatkan
#                     balik jadi Inner. SENGAJA pakai JUMLAH tetangga (bukan cuma jarak ke 1 titik
#                     terdekat) -- titik yg cuma nempel ke SATU titik lain (bisa jadi kebetulan doang)
#                     beda cerita sama titik yg beneran diapit beberapa tiang (lebih meyakinkan dia
#                     bagian dari barisan/jaringan). Ini KEBIJAKAN BISNIS eksplisit, BUKAN soal kepadatan
#                     lagi. Beda sama Layer 3: Layer 3 buat CLUSTER PADAT yg kepencil JAUH (>10km), Layer
#                     4 buat TITIK TUNGGAL/NOISE yg diapit tetangga dalam radius kecil (default 2km).
#                     Set salah satu (atau keduanya) None buat matiin layer ini.
PERSENTIL_OUTLIER_LUNAK = 90
JARAK_ATURAN_BISNIS_KM = 10.0
BATAS_MANDIRI_TERISOLASI = 150
JARAK_RESCUE_DEKAT_KM = 2.0
MIN_TETANGGA_RESCUE = 2            # minimal berapa tiang valid di sekitar supaya dianggap "diapit"
                                    # (bukan cuma nempel ke 1 titik doang). Naikkan buat lebih strict.

K_TETANGGA_CORE_DISTANCE = 10     # k tetangga buat "core distance" -- estimator kepadatan lokal murah
                                   # per titik (dari sklearn NearestNeighbors, bukan hdbscan internal),
                                   # dipakai buat RANKING "inner dari yang paling padat". Makin kecil
                                   # core distance-nya, makin padat lingkungan titik itu.

# analisa dipecah per kolom 'regional', bukan digabung jadi 1 peta besar.
# Root/jangkar, proyeksi UTM, dan kategorisasi dihitung SENDIRI-SENDIRI tiap regional,
# supaya tiang di Jateng gak nyambung/ketimpuk sama tiang di Jatim misalnya.
# MODE_REGIONAL = 'SATU'  -> cuma proses 1 regional yang kamu pilih di REGIONAL_TERPILIH
# MODE_REGIONAL = 'SEMUA' -> loop SEMUA regional sekaligus
MODE_REGIONAL = 'SEMUA'
REGIONAL_TERPILIH = 'R045-JABOJABAR'    # dipakai kalau MODE_REGIONAL = 'SATU'

# kontrol titik awal (seed/jangkar "Main Network"), berdasarkan kolom 'stort'.
# MODE_SEED = 'OTOMATIS'    -> prioritaskan STORT dengan jumlah TIANG PALING BANYAK dulu, baru di
#                               dalam STORT itu dicari titik yang paling padat (core distance terkecil)
#                               sbg jangkarnya. (Kalau kolom 'stort' gak ada, fallback ke titik dengan
#                               core distance terkecil se-regional -- titik terpadat murni dari koordinat.)
# MODE_SEED = 'PILIH_STORT' -> paksa jangkar mulai dari dalam salah satu STORT tertentu (misal 'BYL1'),
#                               lalu di dalam stort itu dicari titik paling padat sbg jangkarnya.
#                               Cluster yang MENGANDUNG titik itu yang jadi "Main Network (Root)".
MODE_SEED = 'OTOMATIS'
SEED_STORT_TERPILIH = 'BYL1'        # dipakai kalau MODE_REGIONAL='SATU' & MODE_SEED='PILIH_STORT'
SEED_STORT_PER_REGIONAL = {         # dipakai kalau MODE_REGIONAL='SEMUA' & MODE_SEED='PILIH_STORT'.
    'R06 JAWA TENGAH': 'BYL1',      # Regional yang gak disebut di sini otomatis fallback ke mode OTOMATIS
    # 'R07 JAWA TIMUR': 'STORT_X',  # buat regional itu aja.
}

# RESTRICTION / BLACKLIST STORT.
# Stort yang didaftar di sini DIKELUARKAN TOTAL dari clustering (dianggap gak ada) sebelum HDBSCAN
# dijalankan -> tiang-tiangnya gak akan pernah jadi Main Network / Sub-Cluster Valid, dan gak bisa
# jadi "jembatan" yang nyambungin 2 cluster lain. Tetap muncul di plot/CSV/peta, tapi statusnya selalu
# "🚫 Dikecualikan (Blacklist Stort)". Cocok buat exclude STORT yang emang di luar scope analisa/proyek
# ini. Pencocokan nama stort case-insensitive & spasi di ujung diabaikan.
STORT_DIKECUALIKAN_TERPILIH = []     # dipakai kalau MODE_REGIONAL='SATU', isi list, misal: ['BYL2', 'BYL3']
STORT_DIKECUALIKAN_PER_REGIONAL = {  # dipakai kalau MODE_REGIONAL='SEMUA'
    # 'R06 JAWA TENGAH': ['STORT_X', 'STORT_Y'],
    # Regional yang gak disebut di sini = gak ada stort yang dikecualikan (proses normal semua).
}

BUAT_PETA_INTERAKTIF = True  # bikin file .html peta OpenStreetMap beneran (folium) per regional
                              # (butuh: !pip install folium -q, dan koneksi internet pas dibuka)
# ==========================================

print("Memuat dataset...")
try:
    df = pd.read_csv(INPUT_FILE)
    df['geometry'] = df['wkt'].apply(loads)
    df['lon'] = df['geometry'].apply(lambda p: p.x)
    df['lat'] = df['geometry'].apply(lambda p: p.y)
except Exception as e:
    print(f"Pastikan file '{INPUT_FILE}' sudah di-upload. Error: {e}")
    import sys; sys.exit()

if 'regional' not in df.columns:
    print("Kolom 'regional' tidak ditemukan di file ini, analisa dijalankan gabungan (1 grup: 'SEMUA').")
    df['regional'] = 'SEMUA'

daftar_regional_tersedia = sorted(df['regional'].dropna().unique())
print(f"Regional yang tersedia di file ini: {daftar_regional_tersedia}")

if MODE_REGIONAL == 'SATU':
    if REGIONAL_TERPILIH not in daftar_regional_tersedia:
        print(f"\n⚠️  REGIONAL_TERPILIH='{REGIONAL_TERPILIH}' tidak ada di data. "
              f"Pilih salah satu dari: {daftar_regional_tersedia}")
        import sys; sys.exit()
    daftar_regional_proses = [REGIONAL_TERPILIH]
else:
    daftar_regional_proses = daftar_regional_tersedia
print(f"Regional yang akan diproses: {daftar_regional_proses}\n")

# --- tentukan titik awal (seed/stort) tiap regional ---
if MODE_SEED == 'PILIH_STORT':
    if MODE_REGIONAL == 'SATU':
        stort_final = {REGIONAL_TERPILIH: SEED_STORT_TERPILIH}
    else:
        stort_final = {reg: SEED_STORT_PER_REGIONAL.get(reg, None) for reg in daftar_regional_proses}
else:
    stort_final = {reg: None for reg in daftar_regional_proses}

print(f"Titik jangkar (seed) tiap regional:")
for reg in daftar_regional_proses:
    print(f"   - {reg}: {stort_final[reg] if stort_final[reg] else 'Otomatis (STORT terbanyak tiangnya)'}")
print()

# --- tentukan stort yang DIKECUALIKAN (blacklist) tiap regional ---
if MODE_REGIONAL == 'SATU':
    stort_dikecualikan_final = {REGIONAL_TERPILIH: [str(s).strip() for s in STORT_DIKECUALIKAN_TERPILIH]}
else:
    key_ngawur_excl = [k for k in STORT_DIKECUALIKAN_PER_REGIONAL if k not in daftar_regional_tersedia]
    if key_ngawur_excl:
        print(f"⚠️  Key di STORT_DIKECUALIKAN_PER_REGIONAL ini gak ketemu di data (dicek typo-nya): {key_ngawur_excl}")
    stort_dikecualikan_final = {
        reg: [str(s).strip() for s in STORT_DIKECUALIKAN_PER_REGIONAL.get(reg, [])]
        for reg in daftar_regional_proses
    }

print(f"STORT yang di-blacklist tiap regional:")
for reg in daftar_regional_proses:
    daftar_excl = stort_dikecualikan_final[reg]
    print(f"   - {reg}: {daftar_excl if daftar_excl else '(tidak ada, semua stort diproses normal)'}")
print()


def epsg_utm_dari_koordinat(lon_arr, lat_arr):
    """Auto-deteksi EPSG UTM (WGS84) dari median lon/lat tiap regional -- 1 zona per regional,
    cukup akurat buat analisa jarak level regional (eps/core-distance langsung kebaca dalam METER),
    meski titik yang pas di pinggir zona (beda 6 derajat bujur) bakal sedikit meleset. Indonesia:
    zona 46-54, belahan bumi selatan -> kode EPSG 327xx (kalau lat >= 0, dianggap 326xx / utara)."""
    center_lon = np.median(lon_arr)
    center_lat = np.median(lat_arr)
    zona = int((center_lon + 180) / 6) + 1
    return (32700 if center_lat < 0 else 32600) + zona


def cari_mask_koordinat_invalid(df_r):
    """Buang koordinat yang jelas rusak SEBELUM clustering -- kalau kebawa, bisa bikin 'cluster'
    palsu yang sangat padat (density artifact), misalnya banyak tiang ke-snap ke titik (0,0)."""
    lon, lat = df_r['lon'].to_numpy(), df_r['lat'].to_numpy()
    return (
        ((lon == 0) & (lat == 0))
        | (lon < 90) | (lon > 145)   # di luar bounding box Indonesia
        | (lat < -12) | (lat > 8)
    )


def cari_seed_point(df_aktif, coords_m, kdtree_m, stort_pilihan, reg_name, radius_densitas_m=500.0):
    """Tentukan index titik jangkar.
    - Kalau stort_pilihan diisi manual (MODE_SEED='PILIH_STORT'): paksa cari di dalam STORT itu.
    - Kalau OTOMATIS (stort_pilihan kosong): prioritaskan STORT dengan jumlah TIANG PALING BANYAK
      dulu, baru di dalam STORT terbanyak itu dicari titik yang paling padat tetangganya sbg jangkar.
    - Kalau kolom 'stort' gak ada sama sekali di data -> return None, caller bakal fallback ke titik
      dengan core distance terkecil se-regional (titik terpadat murni dari koordinat).
    coords_m/kdtree_m dalam METER (hasil proyeksi UTM), bukan derajat lat/lon lagi.
    """
    if 'stort' not in df_aktif.columns:
        if stort_pilihan:
            print(f"      ⚠️  Kolom 'stort' gak ada di data, SEED_STORT diabaikan -> pakai mode otomatis lama.")
        return None, 'OTOMATIS (kolom stort tidak ada, pakai titik ber-core-distance terkecil)'

    if stort_pilihan:
        stort_tersedia = sorted(df_aktif['stort'].dropna().unique())
        cocok = [s for s in stort_tersedia if str(s).strip().lower() == str(stort_pilihan).strip().lower()]
        if not cocok:
            print(f"      ⚠️  SEED_STORT='{stort_pilihan}' gak ketemu di regional '{reg_name}'. "
                  f"Stort yang tersedia di sini: {stort_tersedia}. Fallback ke STORT terbanyak (otomatis).")
            stort_asli = None
            keterangan_awal = None
        else:
            stort_asli = cocok[0]
            keterangan_awal = f"STORT='{stort_asli}' (dipilih manual)"
    else:
        stort_asli = None
        keterangan_awal = None

    if stort_asli is None:
        hitung_stort = df_aktif['stort'].value_counts()
        if len(hitung_stort) == 0:
            return None, 'OTOMATIS (kolom stort kosong semua, pakai titik ber-core-distance terkecil)'
        stort_asli = hitung_stort.idxmax()
        keterangan_awal = f"OTOMATIS: STORT terbanyak='{stort_asli}' ({int(hitung_stort.max())} tiang)"

    idx_kandidat = df_aktif.index[df_aktif['stort'] == stort_asli].to_numpy()
    if len(idx_kandidat) == 0:
        return None, 'OTOMATIS (stort kosong, pakai titik ber-core-distance terkecil)'

    MAKS_KANDIDAT = 5000  # batas biar tetap cepat walau 1 stort isinya puluhan ribu tiang
    if len(idx_kandidat) > MAKS_KANDIDAT:
        idx_kandidat = np.random.RandomState(42).choice(idx_kandidat, MAKS_KANDIDAT, replace=False)

    neighbor_counts = np.array([len(kdtree_m.query_ball_point(coords_m[i], r=radius_densitas_m)) for i in idx_kandidat])
    best_local = int(idx_kandidat[np.argmax(neighbor_counts)])
    return best_local, f"{keterangan_awal} -> titik terpadat di situ"


def cari_mask_diblokir(df_r, stort_dikecualikan, reg_name):
    """Cari mask baris yang di-blacklist -> dikeluarkan TOTAL dari clustering."""
    n = len(df_r)
    mask = np.zeros(n, dtype=bool)
    if not stort_dikecualikan:
        return mask
    if 'stort' not in df_r.columns:
        print(f"      ⚠️  Kolom 'stort' gak ada di data, STORT_DIKECUALIKAN diabaikan.")
        return mask

    stort_tersedia = sorted(df_r['stort'].dropna().unique())
    target_lower = {str(s).strip().lower() for s in stort_dikecualikan}
    stort_cocok = [s for s in stort_tersedia if str(s).strip().lower() in target_lower]
    tidak_ketemu = sorted(target_lower - {str(s).strip().lower() for s in stort_cocok})
    if tidak_ketemu:
        print(f"      ⚠️  STORT_DIKECUALIKAN berikut gak ketemu di regional '{reg_name}' (dicek typo-nya): {tidak_ketemu}")
    if not stort_cocok:
        return mask

    mask = df_r['stort'].isin(stort_cocok).to_numpy()
    print(f"      🚫 STORT dikecualikan: {stort_cocok} -> {int(mask.sum())} tiang diblokir total dari clustering.")
    return mask


def analisa_satu_regional(df_r, stort_pilihan, reg_name, stort_dikecualikan,
                           MIN_CLUSTER_SIZE, MIN_SAMPLES, CLUSTER_SELECTION_EPSILON_M, CLUSTER_SELECTION_METHOD,
                           PERSENTIL_OUTLIER_LUNAK, JARAK_ATURAN_BISNIS_KM, BATAS_MANDIRI_TERISOLASI,
                           K_TETANGGA_CORE_DISTANCE, JARAK_RESCUE_DEKAT_KM=None, MIN_TETANGGA_RESCUE=None):
    """Jalankan HDBSCAN + kategorisasi 4-lapis untuk 1 subset regional saja.

    CATATAN PROYEKSI: jarak dihitung di ruang UTM (meter), BUKAN langsung di derajat lat/lon --
    supaya eps/core-distance/jarak aturan-bisnis semua kebaca dalam satuan yang konsisten & gak
    distorsi antar wilayah (lihat epsg_utm_dari_koordinat).

    CATATAN BLACKLIST & KOORDINAT INVALID: tiang dari STORT yang dikecualikan, dan tiang yang
    koordinatnya jelas rusak ((0,0) / di luar bounding box Indonesia), dikeluarkan SEBELUM HDBSCAN
    dijalankan, jadi gak ikut membentuk cluster / gak bisa jadi density-artifact.

    CATATAN 3 LAPIS "OUTER" (lihat juga catatan parameter di bagian atas file):
    Layer 1 KERAS   = label HDBSCAN == -1 (noise asli, gak nyambung ke kepadatan manapun).
    Layer 2 LUNAK   = skor GLOSH (outlier_scores_) di atas persentil PERSENTIL_OUTLIER_LUNAK, di
                      antara titik yang SUDAH masuk cluster -- masih dihitung Inner tapi ditandai
                      "Soft Outlier / Pinggiran" biar gampang diaudit.
    Layer 3 ATURAN BISNIS = cluster (bukan noise, bukan root) yang jaraknya ke cluster VALID lain
                      manapun > JARAK_ATURAN_BISNIS_KM, KECUALI ukurannya sendiri udah >=
                      BATAS_MANDIRI_TERISOLASI. Ini WAJIB ada di luar HDBSCAN sendiri, karena HDBSCAN
                      cuma peduli KEPADATAN LOKAL -- cluster yang padat secara lokal tapi kepencil
                      jauh dari jaringan padat manapun TETAP dianggap cluster valid oleh HDBSCAN.
    Layer 4 RESCUE - DIAPIT TETANGGA = titik noise (Layer 1) yang DIAPIT (dikelilingi minimal
                      MIN_TETANGGA_RESCUE tiang non-noise) dalam radius JARAK_RESCUE_DEKAT_KM
                      diselamatkan balik jadi Inner. SENGAJA pakai JUMLAH tetangga, bukan cuma jarak
                      ke 1 titik terdekat -- titik yg cuma nempel ke SATU titik (bisa kebetulan doang)
                      beda cerita sama titik yg beneran diapit beberapa tiang di sekitarnya (lebih
                      meyakinkan dia bagian dari barisan/jaringan, bukan noise asli). PENTING: divalidasi
                      manual (lihat catatan parameter) -- titik yang keliatan "nempel" ke area padat
                      di plot skala regional itu SERING kali beneran berjarak ratusan meter-beberapa km
                      (cuma keliatan deket krn plotnya meliputi ratusan km), jadi ini murni KEBIJAKAN
                      BISNIS ("masih diapit dalam radius segini dianggap 1 area"), bukan koreksi
                      kepadatan lagi.
    """
    df_r = df_r.reset_index(drop=True)

    mask_invalid = cari_mask_koordinat_invalid(df_r)
    if mask_invalid.any():
        print(f"      ⚠️  {int(mask_invalid.sum())} tiang koordinatnya aneh ((0,0)/di luar Indonesia) -> dikeluarkan dari clustering.")
    diblokir_mask = cari_mask_diblokir(df_r, stort_dikecualikan, reg_name)

    kategori_keluar = np.full(len(df_r), '', dtype=object)
    kategori_keluar[diblokir_mask] = '🚫 Dikecualikan (Blacklist Stort)'
    kategori_keluar[mask_invalid] = '❌ Koordinat Tidak Valid'  # prioritas kalau tumpang tindih sama blacklist

    mask_keluar = mask_invalid | diblokir_mask
    df_keluar = df_r[mask_keluar].copy()
    df_keluar['Kategori_Propagasi'] = kategori_keluar[mask_keluar]
    df_aktif = df_r[~mask_keluar].reset_index(drop=True)
    n = len(df_aktif)

    if n == 0:
        print(f"      ⚠️  Semua tiang di regional ini kena blacklist/koordinat invalid, gak ada yang diproses.")
        df_keluar['cluster_id'] = -1
        df_keluar['Jumlah_Tiang_Cluster'] = 0
        return df_keluar, None, float('nan')

    # --- proyeksi ke UTM (meter) ---
    coords_deg = df_aktif[['lon', 'lat']].values
    epsg_dipakai = epsg_utm_dari_koordinat(coords_deg[:, 0], coords_deg[:, 1])
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_dipakai}", always_xy=True)
    x_m, y_m = transformer.transform(coords_deg[:, 0], coords_deg[:, 1])
    coords_m = np.column_stack([x_m, y_m])

    # 1. HDBSCAN -- cluster + noise (-1) native
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        cluster_selection_epsilon=float(CLUSTER_SELECTION_EPSILON_M),
        cluster_selection_method=CLUSTER_SELECTION_METHOD,
        metric='euclidean',
        core_dist_n_jobs=-1,
        gen_min_span_tree=True,
    ).fit(coords_m)
    labels = clusterer.labels_
    glosh_scores = clusterer.outlier_scores_
    try:
        dbcv = float(clusterer.relative_validity_)   # DBCV -- bukan silhouette (asumsi cembung, salah
                                                       # utk cluster tiang yg memanjang ngikutin jalan)
    except Exception:
        dbcv = float('nan')

    df_aktif['cluster_id'] = labels
    cluster_sizes_s = pd.Series(labels).value_counts()
    df_aktif['Jumlah_Tiang_Cluster'] = pd.Series(labels).map(cluster_sizes_s).to_numpy()
    df_aktif.loc[labels == -1, 'Jumlah_Tiang_Cluster'] = 0  # noise bukan "cluster", size-nya gak relevan

    # 2. core distance -- estimator kepadatan lokal MURAH per titik, dipakai buat ranking "inner dari
    #    yang paling padat" (bukan hitung ulang lewat HDBSCAN internal)
    k_eff = min(K_TETANGGA_CORE_DISTANCE + 1, n)
    nn = NearestNeighbors(n_neighbors=k_eff).fit(coords_m)
    dist_nn, _ = nn.kneighbors(coords_m)
    core_dist_m = dist_nn[:, -1]
    df_aktif['Core_Distance_m'] = core_dist_m
    df_aktif['Rank_Kepadatan'] = pd.Series(core_dist_m).rank(method='min').astype(int)  # 1 = paling padat
    df_aktif['Skor_Outlier_GLOSH'] = glosh_scores

    # 3. Root/jangkar: OTOMATIS (STORT terbanyak) atau dipaksa dari SEED_STORT
    kdtree_m = KDTree(coords_m)
    best_start_idx, keterangan_seed = cari_seed_point(df_aktif, coords_m, kdtree_m, stort_pilihan, reg_name)
    if best_start_idx is None:
        best_start_idx = int(np.argmin(core_dist_m))  # titik ber-core-distance terkecil se-regional
    root_label = labels[best_start_idx]
    if root_label == -1:
        mask_valid0 = labels != -1
        if mask_valid0.any():
            kdt_valid0 = KDTree(coords_m[mask_valid0])
            _, idx_rel = kdt_valid0.query(coords_m[best_start_idx], k=1)
            root_label = labels[np.where(mask_valid0)[0][idx_rel]]
            print(f"      ⚠️  Titik jangkar kebetulan jatuh di titik noise (HDBSCAN) -> digeser ke cluster valid terdekat.")
    print(f"   🌱 Titik jangkar [{keterangan_seed}] -> cluster label {root_label} | "
          f"DBCV={dbcv:.3f} | {int(labels.max()) + 1} cluster, {int((labels == -1).sum())} noise "
          f"({(labels == -1).mean() * 100:.1f}%)")

    # 4. LAYER 1 (KERAS): noise asli
    is_noise = labels == -1

    # 5. LAYER 3 (ATURAN BISNIS): cluster (bukan noise/root) yang kepencil jauh dari SEMUA cluster
    #    valid lain, kecuali ukurannya sendiri udah gede banget (lihat docstring)
    is_bisnis_outlier = np.zeros(n, dtype=bool)
    if JARAK_ATURAN_BISNIS_KM is not None:
        JARAK_M = JARAK_ATURAN_BISNIS_KM * 1000.0
        for lab, sz in cluster_sizes_s.items():
            if lab == -1 or lab == root_label or sz >= BATAS_MANDIRI_TERISOLASI:
                continue
            idxs = np.where(labels == lab)[0]
            mask_luar = (labels != lab) & (labels != -1)
            if not mask_luar.any():
                continue
            kdt_luar = KDTree(coords_m[mask_luar])
            d, _ = kdt_luar.query(coords_m[idxs], k=1)
            if d.min() > JARAK_M:
                is_bisnis_outlier[idxs] = True
    n_bisnis = int(is_bisnis_outlier.sum())
    if n_bisnis > 0:
        print(f"      📉 Layer 3 (Aturan Bisnis): {n_bisnis} tiang di cluster yang padat SECARA LOKAL "
              f"tapi kepencil > {JARAK_ATURAN_BISNIS_KM} km dari cluster valid manapun -> Outlier.")

    # 6. LAYER 2 (LUNAK): skor GLOSH di atas persentil, cuma di antara titik yg masih calon Inner
    is_soft_outlier = np.zeros(n, dtype=bool)
    kandidat_lunak = (~is_noise) & (~is_bisnis_outlier) & (labels != root_label)
    if PERSENTIL_OUTLIER_LUNAK is not None and kandidat_lunak.any():
        ambang_lunak = np.percentile(glosh_scores[kandidat_lunak], PERSENTIL_OUTLIER_LUNAK)
        is_soft_outlier = kandidat_lunak & (glosh_scores >= ambang_lunak)

    # 7. LAYER 4 (RESCUE - DIAPIT TETANGGA): titik noise yang DIAPIT (dikelilingi >= MIN_TETANGGA_RESCUE
    #    tiang non-noise dalam radius JARAK_RESCUE_DEKAT_KM) diselamatkan balik jadi Inner. SENGAJA pakai
    #    JUMLAH tetangga, bukan cuma jarak ke 1 titik terdekat -- soalnya titik yg cuma nempel ke SATU
    #    titik lain (mungkin kebetulan doang) beda cerita sama titik yg beneran "diapit" tiang di sekitarnya
    #    (lebih meyakinkan itu bagian dari barisan/jaringan, bukan noise asli). KEBIJAKAN BISNIS eksplisit,
    #    lihat catatan di docstring & parameter.
    is_rescued = np.zeros(n, dtype=bool)
    if JARAK_RESCUE_DEKAT_KM is not None and MIN_TETANGGA_RESCUE is not None and is_noise.any():
        mask_acuan_rescue = ~is_noise  # semua yg BUKAN noise = acuan "masih deket jaringan"
        if mask_acuan_rescue.any():
            kdt_acuan = KDTree(coords_m[mask_acuan_rescue])
            idx_noise = np.where(is_noise)[0]
            JARAK_RESCUE_M = JARAK_RESCUE_DEKAT_KM * 1000.0
            jumlah_tetangga = np.array([
                len(kdt_acuan.query_ball_point(coords_m[i], r=JARAK_RESCUE_M)) for i in idx_noise
            ])
            is_rescued[idx_noise[jumlah_tetangga >= MIN_TETANGGA_RESCUE]] = True
    n_rescue = int(is_rescued.sum())
    if n_rescue > 0:
        print(f"      🔎 Layer 4 (Rescue - Diapit Tetangga): {n_rescue} tiang noise yang ternyata "
              f"DIAPIT >= {MIN_TETANGGA_RESCUE} tiang valid dalam radius {JARAK_RESCUE_DEKAT_KM} km "
              f"-> diselamatkan jadi Inner.")

    # 8. Gabungin jadi Kategori_Propagasi final
    kategori = np.where(
        labels == root_label, 'Main Network (Root)',
        np.where(is_rescued, 'Sub-Cluster Valid (Rescue - Dekat Cluster)',
        np.where(is_noise, 'Outlier (Noise - HDBSCAN)',
        np.where(is_bisnis_outlier, 'Outlier (Aturan Bisnis - Terisolasi)',
        np.where(is_soft_outlier, 'Sub-Cluster Valid (Soft Outlier - Pinggiran)',
        'Sub-Cluster Valid'))))
    )
    df_aktif['Kategori_Propagasi'] = kategori

    root_point = coords_deg[best_start_idx].copy()  # simpan KOORDINAT lon/lat asli (bukan meter/index)

    if len(df_keluar) > 0:
        df_keluar['cluster_id'] = -1
        df_keluar['Jumlah_Tiang_Cluster'] = 0
        df_gabung = pd.concat([df_aktif, df_keluar], ignore_index=True)
    else:
        df_gabung = df_aktif

    return df_gabung, root_point, dbcv


hasil_per_regional = []
info_root_point = {}
info_dbcv = {}
for reg in daftar_regional_proses:
    df_r = df[df['regional'] == reg].copy()
    print(f"📍 Regional: {reg}  (total {len(df_r)} tiang)")
    df_r, root_point, dbcv = analisa_satu_regional(
        df_r, stort_final[reg], reg, stort_dikecualikan_final[reg],
        MIN_CLUSTER_SIZE, MIN_SAMPLES, CLUSTER_SELECTION_EPSILON_M, CLUSTER_SELECTION_METHOD,
        PERSENTIL_OUTLIER_LUNAK, JARAK_ATURAN_BISNIS_KM, BATAS_MANDIRI_TERISOLASI, K_TETANGGA_CORE_DISTANCE,
        JARAK_RESCUE_DEKAT_KM, MIN_TETANGGA_RESCUE
    )
    # cluster_id dibikin unik lintas regional biar gak ketuker pas digabung nanti
    df_r['cluster_id'] = reg + "_" + df_r['cluster_id'].astype(str)

    n_main = (df_r['Kategori_Propagasi'] == 'Main Network (Root)').sum()
    n_sub = (df_r['Kategori_Propagasi'] == 'Sub-Cluster Valid').sum()
    n_lunak = (df_r['Kategori_Propagasi'] == 'Sub-Cluster Valid (Soft Outlier - Pinggiran)').sum()
    n_rescue = (df_r['Kategori_Propagasi'] == 'Sub-Cluster Valid (Rescue - Dekat Cluster)').sum()
    n_bisnis = (df_r['Kategori_Propagasi'] == 'Outlier (Aturan Bisnis - Terisolasi)').sum()
    n_noise = (df_r['Kategori_Propagasi'] == 'Outlier (Noise - HDBSCAN)').sum()
    n_diblokir = (df_r['Kategori_Propagasi'] == '🚫 Dikecualikan (Blacklist Stort)').sum()
    n_invalid = (df_r['Kategori_Propagasi'] == '❌ Koordinat Tidak Valid').sum()
    n_sub_clusters = df_r.loc[df_r['Kategori_Propagasi'].isin(
        ['Sub-Cluster Valid', 'Sub-Cluster Valid (Soft Outlier - Pinggiran)']), 'cluster_id'].nunique()

    print(f"   -> Main Network (Root)                        : {n_main} tiang")
    print(f"   -> Sub-Cluster Valid (>= {MIN_CLUSTER_SIZE} tiang)             : {n_sub} tiang ({n_sub_clusters} cluster)")
    if n_lunak > 0:
        print(f"   -> Sub-Cluster Valid (Soft Outlier / Pinggiran) : {n_lunak} tiang "
              f"(GLOSH >= persentil {PERSENTIL_OUTLIER_LUNAK}, tetap Inner tapi ditandai)")
    if n_rescue > 0:
        print(f"   -> Sub-Cluster Valid (Rescue - Dekat Cluster)   : {n_rescue} tiang "
              f"(noise yg diselamatkan, diapit >= {MIN_TETANGGA_RESCUE} tiang dlm radius {JARAK_RESCUE_DEKAT_KM} km)")
    print(f"   -> Outlier (Aturan Bisnis - Terisolasi)       : {n_bisnis} tiang")
    print(f"   -> Outlier (Noise - HDBSCAN)                  : {n_noise} tiang")
    if n_invalid > 0:
        print(f"   -> ❌ Koordinat Tidak Valid                    : {n_invalid} tiang")
    if n_diblokir > 0:
        print(f"   -> 🚫 Dikecualikan (Blacklist Stort)           : {n_diblokir} tiang")
    print()

    info_root_point[reg] = root_point
    info_dbcv[reg] = dbcv
    hasil_per_regional.append(df_r)

df_all = pd.concat(hasil_per_regional, ignore_index=True)

# ==========================================
# 4. VISUALISASI: 1 PANEL PER REGIONAL
# ==========================================
n_reg = len(daftar_regional_proses)
n_cols = 2 if n_reg > 1 else 1
n_rows = int(np.ceil(n_reg / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 7 * n_rows), squeeze=False)
warna = {
    'Main Network (Root)': '#3498db',
    'Sub-Cluster Valid': '#2ecc71',
    'Sub-Cluster Valid (Soft Outlier - Pinggiran)': '#f39c12',
    'Sub-Cluster Valid (Rescue - Dekat Cluster)': '#1abc9c',
    'Outlier (Aturan Bisnis - Terisolasi)': '#9b59b6',
    'Outlier (Noise - HDBSCAN)': '#e74c3c',
    '❌ Koordinat Tidak Valid': '#2c3e50',
    '🚫 Dikecualikan (Blacklist Stort)': '#95a5a6',
}
ukuran = {
    'Main Network (Root)': 15, 'Sub-Cluster Valid': 22,
    'Sub-Cluster Valid (Soft Outlier - Pinggiran)': 22, 'Sub-Cluster Valid (Rescue - Dekat Cluster)': 22,
    'Outlier (Aturan Bisnis - Terisolasi)': 28,
    'Outlier (Noise - HDBSCAN)': 25, '❌ Koordinat Tidak Valid': 30, '🚫 Dikecualikan (Blacklist Stort)': 18,
}
KAT_ATAS = ('Outlier (Aturan Bisnis - Terisolasi)', 'Outlier (Noise - HDBSCAN)',
            '❌ Koordinat Tidak Valid', '🚫 Dikecualikan (Blacklist Stort)')

for i, reg in enumerate(daftar_regional_proses):
    ax = axes[i // n_cols][i % n_cols]
    df_r = df_all[df_all['regional'] == reg]
    for kat, c in warna.items():
        sub = df_r[df_r['Kategori_Propagasi'] == kat]
        if len(sub) == 0:
            continue
        ax.scatter(sub['lon'], sub['lat'], c=c, s=ukuran[kat], alpha=0.85,
                   edgecolors='black' if kat != 'Main Network (Root)' else 'none',
                   linewidths=0.5, label=f'{kat} ({len(sub)})',
                   zorder=3 if kat in KAT_ATAS else 2)
    root_point = info_root_point.get(reg)
    if root_point is not None:
        ax.scatter(root_point[0], root_point[1], c='#f1c40f', marker='*', s=350,
                   edgecolors='black', label='Titik Jangkar (Root)', zorder=4)
    dbcv = info_dbcv.get(reg, float('nan'))
    ax.set_title(f'{reg}\nHDBSCAN min_cluster={MIN_CLUSTER_SIZE} min_samples={MIN_SAMPLES} '
                 f'eps={CLUSTER_SELECTION_EPSILON_M:.0f}m | DBCV={dbcv:.3f}',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend(loc='upper right', framealpha=0.9, fontsize=7)
    ax.grid(True, linestyle=':', alpha=0.7)

# matikan panel kosong kalau jumlah regional ganjil
for j in range(n_reg, n_rows * n_cols):
    axes[j // n_cols][j % n_cols].axis('off')

plt.tight_layout()
plt.show()

# ==========================================
# 5. EXPORT CSV (semua regional, 1 file, kolom 'regional' jadi penanda)
# ==========================================
df_export = df_all.drop(columns=['geometry', 'lon', 'lat'])
export_filename = f"TiangJabo_HDBSCAN_mcs{MIN_CLUSTER_SIZE}_ms{MIN_SAMPLES}_eps{int(CLUSTER_SELECTION_EPSILON_M)}m_per_regional.csv"
df_export.to_csv(export_filename, index=False)
print(f"✅ File CSV berhasil disimpan: {export_filename}")
print("   (kolom 'Core_Distance_m' & 'Rank_Kepadatan' bisa dipakai buat urutin tiang dari yang paling")
print("    padat -- makin kecil Core_Distance_m / makin kecil Rank_Kepadatan, makin padat lingkungannya)")

# ==========================================
# 6. PETA INTERAKTIF (folium) - basemap OpenStreetMap asli per regional
# ==========================================
if BUAT_PETA_INTERAKTIF:
    try:
        import folium

        MAKS_TITIK_PER_KATEGORI = 4000  # batas titik yg digambar 1-1 per kategori biar peta tetap ringan
        warna_hex = dict(warna)

        for reg in daftar_regional_proses:
            df_r = df_all[df_all['regional'] == reg]
            if len(df_r) == 0:
                continue
            pusat_lat, pusat_lon = df_r['lat'].mean(), df_r['lon'].mean()
            m = folium.Map(location=[pusat_lat, pusat_lon], zoom_start=9, tiles='OpenStreetMap')

            for kat, warna_k in warna_hex.items():
                sub = df_r[df_r['Kategori_Propagasi'] == kat]
                if len(sub) == 0:
                    continue
                catatan = ""
                if len(sub) > MAKS_TITIK_PER_KATEGORI:
                    sub = sub.sample(MAKS_TITIK_PER_KATEGORI, random_state=42)
                    catatan = f" (sampel {MAKS_TITIK_PER_KATEGORI} dari {len(df_r[df_r['Kategori_Propagasi']==kat])})"
                fg = folium.FeatureGroup(name=f'{kat} ({len(df_r[df_r["Kategori_Propagasi"]==kat])}){catatan}')
                for _, row in sub.iterrows():
                    folium.CircleMarker(
                        location=[row['lat'], row['lon']], radius=3, color=warna_k,
                        fill=True, fill_color=warna_k, fill_opacity=0.8, weight=1,
                        popup=str(row['name']) if 'name' in row else None,
                    ).add_to(fg)
                fg.add_to(m)

            root_point = info_root_point.get(reg)
            if root_point is not None:
                folium.Marker(
                    location=[root_point[1], root_point[0]],
                    icon=folium.Icon(color='orange', icon='star'),
                    popup='Titik Jangkar (Root)',
                ).add_to(m)

            folium.LayerControl(collapsed=False).add_to(m)
            nama_file_peta = f"Peta_HDBSCAN_{reg.replace(' ', '_')}.html"
            m.save(nama_file_peta)
            print(f"🗺️  Peta interaktif disimpan: {nama_file_peta} (buka di browser buat ngecek visual)")
    except ImportError:
        print("\n⚠️  Package 'folium' belum ke-install, peta interaktif dilewati.")
        print("    Jalankan `!pip install folium -q` di cell terpisah lalu run ulang script ini kalau mau peta-nya.")

# ==========================================
# CATATAN LANJUTAN (gak diimplementasi di sini, tapi bisa dipakai kalau butuh):
# - Data jutaan tiang & harus jalan rutin tiap hari -> HDBSCAN bisa berat. Pertimbangkan indeks H3
#   (resolusi 8-9) buat pra-penyaringan wilayah "hot" dulu, baru HDBSCAN detail di dalamnya.
# - Ada tiang BARU & gak mau fit ulang semua (ID cluster bisa berubah total kalau di-fit ulang)
#   -> pakai hdbscan.approximate_predict(clusterer, titik_baru) buat nempelin ke cluster yang sudah ada.
# ==========================================
