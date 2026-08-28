import pandas as pd
import numpy as np
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
INPUT_FILE = "data_tiang.csv"     # nama file csv tiang yang sudah di-upload
RADIUS_SCAN_KM = 1.0              # Jarak maksimal loncatan antar tiang/cluster (1.0 = 1 kilometer)
MIN_TIANG_SUBCLUSTER = 5          # minimal jumlah tiang supaya sekumpulan tiang dianggap
                                   # "sub-cluster valid" (network kecil yg tetap solid), bukan outlier acak

# BARU: koreksi tahap-2 pakai TEKNIK PENCARIAN JARAK (KDTree) ala Code 2 (inner_outer_choose.py).
# Ada 2 masalah yang diperbaiki di sini (ketauan pas ngecek manual hasil plot R045-JABOJABAR):
#
#  MASALAH A - tiang "nempel" ke area padat malah kecap Outlier:
#  connected-components dgn radius KAKU (RADIUS_SCAN_KM) bisa motong sekumpulan tiang jadi komponen
#  kecil (< MIN_TIANG_SUBCLUSTER) walau posisinya masih MENEMPEL/dekat ke area padat (Main Network /
#  Sub-Cluster Valid). Fix: tiang yang divonis Outlier dicek ULANG jaraknya (KDTree) ke tiang padat
#  TERDEKAT. Kalau jaraknya <= JARAK_DEKAT_PADAT_KM -> direklasifikasi jadi Inner.
#
#  MASALAH B - komponen yg KEBETULAN >= MIN_TIANG_SUBCLUSTER tapi BENERAN KEPENCIL SENDIRIAN:
#  sebaliknya, MIN_TIANG_SUBCLUSTER doang gak cukup -- komponen yg padat SECARA LOKAL (misal 59 tiang
#  numpuk rapet) tapi lokasinya keliatan dari SEMUA jaringan lain (puluhan km, gak nempel kemana-mana)
#  sebelumnya lolos otomatis jadi "Sub-Cluster Valid" padahal ini persis "outlier" yg dimaksud: terisolasi
#  DAN jauh dari pusat tiang-tiang padat manapun. Fix: komponen yg jaraknya ke SEMUA komponen lain lebih
#  jauh dari JARAK_TERISOLASI_KM diturunin ke Outlier, KECUALI ukurannya sendiri udah >= BATAS_MANDIRI_TERISOLASI
#  (segede itu, dianggap emang jaringan/kota lain yg sah, walau posisinya jauh sendirian).
#
# Set JARAK_DEKAT_PADAT_KM None/<=RADIUS_SCAN_KM buat matiin Masalah A, atau JARAK_TERISOLASI_KM None
# buat matiin Masalah B (balik ke perilaku lama, murni radius kaku + MIN_TIANG_SUBCLUSTER).
JARAK_DEKAT_PADAT_KM = 5.0        # biasanya diisi LEBIH BESAR dari RADIUS_SCAN_KM (nyari lebih jauh
                                   # drpd radius connectivity, itu intinya) -- mainkan sesuai kepadatan datamu
JARAK_TERISOLASI_KM = 10.0        # di atas jarak ini ke komponen LAIN manapun -> dianggap "beneran kepencil"
BATAS_MANDIRI_TERISOLASI = 150    # kecuali ukuran komponennya sendiri >= ini -> tetep Sub-Cluster Valid

# analisa dipecah per kolom 'regional', bukan digabung jadi 1 peta besar.
# Root/jangkar, radius propagasi, dan kategorisasi dihitung SENDIRI-SENDIRI tiap regional,
# supaya tiang di Jateng gak nyambung/ketimpuk sama tiang di Jatim misalnya.
# MODE_REGIONAL = 'SATU'  -> cuma proses 1 regional yang kamu pilih di REGIONAL_TERPILIH
# MODE_REGIONAL = 'SEMUA' -> loop SEMUA regional sekaligus
MODE_REGIONAL = 'SEMUA'
REGIONAL_TERPILIH = 'R045-JABOJABAR'    # dipakai kalau MODE_REGIONAL = 'SATU'

# BARU: kontrol titik awal (seed/jangkar "Main Network"), berdasarkan kolom 'stort'.
# MODE_SEED = 'OTOMATIS'    -> BARU: prioritaskan STORT dengan jumlah TIANG PALING BANYAK dulu,
#                               baru di dalam STORT itu dicari titik yang paling padat sbg jangkarnya.
#                               (Kalau kolom 'stort' gak ada di data, fallback ke titik terpadat
#                               dekat pusat geografis regional, kaya versi sebelumnya.)
# MODE_SEED = 'PILIH_STORT' -> paksa jangkar mulai dari dalam salah satu STORT tertentu (misal 'BYL1'),
#                               lalu di dalam stort itu dicari titik paling padat sbg jangkarnya.
#                               Cluster yang MENGANDUNG titik itu yang jadi "Main Network (Root)".
MODE_SEED = 'OTOMATIS'
SEED_STORT_TERPILIH = 'BYL1'        # dipakai kalau MODE_REGIONAL='SATU' & MODE_SEED='PILIH_STORT'
SEED_STORT_PER_REGIONAL = {         # dipakai kalau MODE_REGIONAL='SEMUA' & MODE_SEED='PILIH_STORT'.
    'R06 JAWA TENGAH': 'BYL1',      # Regional yang gak disebut di sini otomatis fallback ke mode OTOMATIS
    # 'R07 JAWA TIMUR': 'STORT_X',  # buat regional itu aja.
}

# BARU: RESTRICTION / BLACKLIST STORT.
# Stort yang didaftar di sini DIKELUARKAN TOTAL dari graf konektivitas (dianggap gak ada) sebelum
# connected-components dihitung -> tiang-tiangnya gak akan pernah jadi Main Network / Sub-Cluster Valid,
# dan gak bisa jadi "jembatan" yang nyambungin 2 cluster lain. Tetap muncul di plot/CSV/peta, tapi
# statusnya selalu "🚫 Dikecualikan (Blacklist Stort)". Cocok buat exclude STORT yang emang di luar
# scope analisa/proyek ini. Pencocokan nama stort case-insensitive & spasi di ujung diabaikan.
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


def cari_seed_point(df_aktif, coords, kdtree, stort_pilihan, reg_name):
    """Tentukan index titik jangkar.
    - Kalau stort_pilihan diisi manual (MODE_SEED='PILIH_STORT'): paksa cari di dalam STORT itu.
    - Kalau OTOMATIS (stort_pilihan kosong): BARU -> prioritaskan STORT dengan jumlah TIANG PALING
      BANYAK dulu (bukan cuma titik terdekat ke pusat geografis), baru di dalam STORT terbanyak itu
      dicari titik yang paling padat tetangganya sbg jangkar. Konsisten sama logika Code 2 (yang
      otomatis mulai dari span paling padat).
    - Kalau kolom 'stort' gak ada sama sekali di data -> return None, caller bakal fallback ke cara
      lama (titik terpadat dekat pusat geografis regional, murni dari koordinat).
    """
    if 'stort' not in df_aktif.columns:
        if stort_pilihan:
            print(f"      ⚠️  Kolom 'stort' gak ada di data, SEED_STORT diabaikan -> pakai mode otomatis lama.")
        return None, 'OTOMATIS (kolom stort tidak ada, pakai titik terpadat dekat pusat)'

    if stort_pilihan:
        stort_tersedia = sorted(df_aktif['stort'].dropna().unique())
        cocok = [s for s in stort_tersedia if str(s).strip().lower() == str(stort_pilihan).strip().lower()]
        if not cocok:
            print(f"      ⚠️  SEED_STORT='{stort_pilihan}' gak ketemu di regional '{reg_name}'. "
                  f"Stort yang tersedia di sini: {stort_tersedia}. Fallback ke STORT terbanyak (otomatis).")
            stort_asli = None  # jatuh ke logika 'terbanyak' di bawah
            keterangan_awal = None
        else:
            stort_asli = cocok[0]
            keterangan_awal = f"STORT='{stort_asli}' (dipilih manual)"
    else:
        stort_asli = None
        keterangan_awal = None

    if stort_asli is None:
        # BARU: OTOMATIS = prioritaskan STORT dengan jumlah tiang PALING BANYAK
        hitung_stort = df_aktif['stort'].value_counts()
        if len(hitung_stort) == 0:
            return None, 'OTOMATIS (kolom stort kosong semua, pakai titik terpadat dekat pusat)'
        stort_asli = hitung_stort.idxmax()
        keterangan_awal = f"OTOMATIS: STORT terbanyak='{stort_asli}' ({int(hitung_stort.max())} tiang)"

    idx_kandidat = df_aktif.index[df_aktif['stort'] == stort_asli].to_numpy()
    if len(idx_kandidat) == 0:
        return None, 'OTOMATIS (stort kosong, pakai titik terpadat dekat pusat)'

    MAKS_KANDIDAT = 5000  # batas biar tetap cepat walau 1 stort isinya puluhan ribu tiang
    if len(idx_kandidat) > MAKS_KANDIDAT:
        idx_kandidat = np.random.RandomState(42).choice(idx_kandidat, MAKS_KANDIDAT, replace=False)

    neighbor_counts = np.array([len(kdtree.query_ball_point(coords[i], r=0.5 / 111.0)) for i in idx_kandidat])
    best_local = int(idx_kandidat[np.argmax(neighbor_counts)])
    return best_local, f"{keterangan_awal} -> titik terpadat di situ"


def cari_mask_diblokir(df_r, stort_dikecualikan, reg_name):
    """Cari mask baris yang di-blacklist -> dikeluarkan TOTAL dari graf konektivitas."""
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
    print(f"      🚫 STORT dikecualikan: {stort_cocok} -> {int(mask.sum())} tiang diblokir total dari analisa konektivitas.")
    return mask


def analisa_satu_regional(df_r, RADIUS_SCAN_KM, MIN_TIANG_SUBCLUSTER, stort_pilihan, reg_name, stort_dikecualikan,
                           JARAK_DEKAT_PADAT_KM=None, JARAK_TERISOLASI_KM=None, BATAS_MANDIRI_TERISOLASI=150):
    """Jalankan seluruh logika propagasi & kategorisasi untuk 1 subset regional saja.

    CATATAN PERFORMA: connected-components dibangun dari pasangan tetangga SATU ARAH saja
    (row=pairs[:,0], col=pairs[:,1]) -- scipy sudah otomatis anggap graf undirected kalau
    directed=False, jadi gak perlu disimetriskan manual (dobelin array). Ini terbukti ~6x lebih
    cepat di data padat (10 detik -> 1.7 detik utk 67rb titik dgn 19 juta pasangan tetangga).
    Kategorisasi juga divektorisasi pakai np.where, bukan df.apply(axis=1) yang lambat di data besar.

    CATATAN BLACKLIST: tiang dari STORT yang dikecualikan dikeluarkan SEBELUM KDTree/graf dibangun,
    jadi mereka gak bisa jadi "jembatan" yang nyambungin 2 cluster yang harusnya terpisah.

    CATATAN KOREKSI (ala Code 2 / inner_outer_choose.py) -- 2 tahap, lihat juga catatan parameter
    JARAK_DEKAT_PADAT_KM / JARAK_TERISOLASI_KM / BATAS_MANDIRI_TERISOLASI di bagian atas file:

    Tahap 3a (turunin komponen yg KEPENCIL SENDIRIAN): komponen yg >= MIN_TIANG_SUBCLUSTER (jadi
    lolos jadi 'Sub-Cluster Valid' di step 3) tapi jaraknya ke SEMUA komponen lain lebih jauh dari
    JARAK_TERISOLASI_KM diturunin jadi 'Outlier (Terputus)' -- KECUALI ukurannya sendiri udah
    >= BATAS_MANDIRI_TERISOLASI. Ini dicek pakai graf connected-components KEDUA yg radiusnya jauh
    lebih gede (JARAK_TERISOLASI_KM) drpd RADIUS_SCAN_KM: kalau komponen si tiang di graf-gede ini
    ukurannya SAMA PERSIS kayak komponen di graf-kecil (RADIUS_SCAN_KM), artinya emang gak ada
    komponen lain manapun yg nyambung/deket bahkan di radius segede itu -> beneran kepencil sendirian.

    Tahap 3b (selamatin komponen yg NEMPEL ke area padat): kebalikannya -- komponen kecil (< MIN_TIANG_
    SUBCLUSTER, atau abis diturunin di 3a) yg ternyata masih deket (<= JARAK_DEKAT_PADAT_KM, dicek pakai
    KDTree.query, teknik pencarian jarak-terdekat yg sama kayak dipakai nyari span di Code 2) ke tiang
    padat (Main Network / Sub-Cluster Valid) TERDEKAT, diselamatkan balik jadi Inner.

    Intinya niru definisi "outlier beneran" = terisolasi DAN jauh dari pusat tiang padat manapun
    (3a) DAN dikit tiangnya (gagal MIN_TIANG_SUBCLUSTER atau BATAS_MANDIRI_TERISOLASI) -- tapi begitu
    dia masih nempel ke area padat (3b), tetap dianggap Inner.
    """
    df_r = df_r.reset_index(drop=True)

    diblokir_mask = cari_mask_diblokir(df_r, stort_dikecualikan, reg_name)
    df_diblokir = df_r[diblokir_mask].copy()
    df_aktif = df_r[~diblokir_mask].reset_index(drop=True)
    n = len(df_aktif)

    if n == 0:
        print(f"      ⚠️  Semua tiang di regional ini kena blacklist, gak ada yang diproses.")
        if len(df_diblokir) > 0:
            df_diblokir['cluster_id'] = 'DIBLOKIR'
            df_diblokir['Jumlah_Tiang_Cluster'] = 0
            df_diblokir['Kategori_Propagasi'] = '🚫 Dikecualikan (Blacklist Stort)'
        return df_diblokir, None

    coords = df_aktif[['lon', 'lat']].values
    kdtree = KDTree(coords)
    RADIUS_DEG = RADIUS_SCAN_KM / 111.0

    # 1. Semua connected component dalam regional ini (dioptimasi, lihat catatan performa di atas)
    pairs = kdtree.query_pairs(r=RADIUS_DEG, output_type='ndarray')
    if len(pairs) > 0:
        graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    else:
        graph = coo_matrix((n, n))
    n_components, labels = connected_components(csgraph=graph, directed=False)
    df_aktif['cluster_id'] = labels
    cluster_sizes = df_aktif['cluster_id'].value_counts()
    df_aktif['Jumlah_Tiang_Cluster'] = df_aktif['cluster_id'].map(cluster_sizes).to_numpy()

    # 2. Root/jangkar: OTOMATIS atau dipaksa dari SEED_STORT
    best_start_idx, keterangan_seed = cari_seed_point(df_aktif, coords, kdtree, stort_pilihan, reg_name)
    if best_start_idx is None:
        center_lon, center_lat = df_aktif['lon'].mean(), df_aktif['lat'].mean()
        dists_to_center = np.sum((coords - [center_lon, center_lat]) ** 2, axis=1)
        closest_100_idx = np.argsort(dists_to_center)[:min(100, n)]
        best_start_idx, max_neighbors = -1, -1
        for idx in closest_100_idx:
            n_count = len(kdtree.query_ball_point(coords[idx], r=0.5 / 111.0))
            if n_count > max_neighbors:
                max_neighbors, best_start_idx = n_count, idx
    root_cluster_id = df_aktif.loc[best_start_idx, 'cluster_id']
    print(f"   🌱 Titik jangkar [{keterangan_seed}] -> cluster_id lokal {root_cluster_id}")

    # 3. Kategorisasi -- divektorisasi (np.where), BUKAN df.apply(axis=1) yang lambat di data besar
    cluster_id_arr = df_aktif['cluster_id'].to_numpy()
    jml_cluster_arr = df_aktif['Jumlah_Tiang_Cluster'].to_numpy()
    df_aktif['Kategori_Propagasi'] = np.where(
        cluster_id_arr == root_cluster_id, 'Main Network (Root)',
        np.where(jml_cluster_arr >= MIN_TIANG_SUBCLUSTER, 'Sub-Cluster Valid', 'Outlier (Terputus)')
    )

    # 3a. TURUNIN komponen yang KEPENCIL SENDIRIAN, lihat catatan di docstring.
    # Cuma jalan kalau JARAK_TERISOLASI_KM diisi & lebih besar dari RADIUS_SCAN_KM.
    if JARAK_TERISOLASI_KM is not None and JARAK_TERISOLASI_KM > RADIUS_SCAN_KM:
        JARAK_TERISOLASI_DEG = JARAK_TERISOLASI_KM / 111.0
        pairs_isolasi = kdtree.query_pairs(r=JARAK_TERISOLASI_DEG, output_type='ndarray')
        if len(pairs_isolasi) > 0:
            graph_isolasi = coo_matrix(
                (np.ones(len(pairs_isolasi)), (pairs_isolasi[:, 0], pairs_isolasi[:, 1])), shape=(n, n)
            )
        else:
            graph_isolasi = coo_matrix((n, n))
        _, labels_isolasi = connected_components(csgraph=graph_isolasi, directed=False)
        pulau_sizes = pd.Series(labels_isolasi).value_counts()
        jml_pulau_arr = pd.Series(labels_isolasi).map(pulau_sizes).to_numpy()

        kategori_arr = df_aktif['Kategori_Propagasi'].to_numpy()
        # "kepencil sendirian" = ukuran pulau (radius jauh) SAMA PERSIS kayak ukuran cluster (radius
        # kecil) -> gak ada komponen lain manapun yang nyambung bahkan di radius sejauh itu.
        mask_kepencil = (
            (kategori_arr == 'Sub-Cluster Valid')
            & (jml_pulau_arr == jml_cluster_arr)
            & (jml_cluster_arr < BATAS_MANDIRI_TERISOLASI)
        )
        n_diturunkan = int(mask_kepencil.sum())
        if n_diturunkan > 0:
            kategori_arr = kategori_arr.copy()
            kategori_arr[mask_kepencil] = 'Outlier (Terputus)'
            df_aktif['Kategori_Propagasi'] = kategori_arr
            print(f"      📉 Koreksi isolasi (ala Code 2): {n_diturunkan} tiang yang tadinya "
                  f"'Sub-Cluster Valid' ternyata > {JARAK_TERISOLASI_KM} km dari komponen manapun & "
                  f"< {BATAS_MANDIRI_TERISOLASI} tiang -> diturunkan jadi Outlier (kepencil sendirian).")

    # 3b. SELAMATIN komponen yang NEMPEL ke area padat -- teknik pencarian jarak (KDTree) ala Code 2,
    # lihat catatan di docstring.
    # Hanya jalan kalau JARAK_DEKAT_PADAT_KM diisi & lebih besar dari RADIUS_SCAN_KM (kalau lebih
    # kecil/sama, gak akan pernah nemu apa-apa karena titik segitu deketnya udah pasti KESAMBUNG
    # dari step 1 & gak akan pernah kecap Outlier duluan).
    if JARAK_DEKAT_PADAT_KM is not None and JARAK_DEKAT_PADAT_KM > RADIUS_SCAN_KM:
        kategori_arr = df_aktif['Kategori_Propagasi'].to_numpy()
        mask_padat = np.isin(kategori_arr, ['Main Network (Root)', 'Sub-Cluster Valid'])
        mask_outlier_awal = kategori_arr == 'Outlier (Terputus)'

        if mask_padat.any() and mask_outlier_awal.any():
            kdtree_padat = KDTree(coords[mask_padat])
            JARAK_DEKAT_DEG = JARAK_DEKAT_PADAT_KM / 111.0
            idx_outlier = np.where(mask_outlier_awal)[0]
            dist_ke_padat, _ = kdtree_padat.query(coords[idx_outlier], k=1)
            dekat_ke_padat = dist_ke_padat <= JARAK_DEKAT_DEG

            n_diselamatkan = int(dekat_ke_padat.sum())
            if n_diselamatkan > 0:
                kategori_arr = kategori_arr.copy()
                kategori_arr[idx_outlier[dekat_ke_padat]] = 'Sub-Cluster Valid (Dekat Jaringan Padat)'
                df_aktif['Kategori_Propagasi'] = kategori_arr
                print(f"      🔎 Koreksi jarak (ala Code 2): {n_diselamatkan} tiang yang tadinya "
                      f"'Outlier (Terputus)' ternyata masih <= {JARAK_DEKAT_PADAT_KM} km dari jaringan "
                      f"padat -> direklasifikasi jadi Inner ('Sub-Cluster Valid (Dekat Jaringan Padat)').")

    root_point = coords[best_start_idx].copy()  # simpan KOORDINAT (bukan index, karena index berubah pas digabung)

    if len(df_diblokir) > 0:
        df_diblokir['cluster_id'] = 'DIBLOKIR'
        df_diblokir['Jumlah_Tiang_Cluster'] = 0
        df_diblokir['Kategori_Propagasi'] = '🚫 Dikecualikan (Blacklist Stort)'
        df_gabung = pd.concat([df_aktif, df_diblokir], ignore_index=True)
    else:
        df_gabung = df_aktif

    return df_gabung, root_point


hasil_per_regional = []
info_root_point = {}
for reg in daftar_regional_proses:
    df_r = df[df['regional'] == reg].copy()
    print(f"📍 Regional: {reg}  (total {len(df_r)} tiang)")
    df_r, root_point = analisa_satu_regional(
        df_r, RADIUS_SCAN_KM, MIN_TIANG_SUBCLUSTER, stort_final[reg], reg, stort_dikecualikan_final[reg],
        JARAK_DEKAT_PADAT_KM, JARAK_TERISOLASI_KM, BATAS_MANDIRI_TERISOLASI
    )
    # cluster_id dibikin unik lintas regional biar gak ketuker pas digabung nanti
    df_r['cluster_id'] = reg + "_" + df_r['cluster_id'].astype(str)

    n_main = (df_r['Kategori_Propagasi'] == 'Main Network (Root)').sum()
    n_sub = (df_r['Kategori_Propagasi'] == 'Sub-Cluster Valid').sum()
    n_dekat_padat = (df_r['Kategori_Propagasi'] == 'Sub-Cluster Valid (Dekat Jaringan Padat)').sum()
    n_outlier = (df_r['Kategori_Propagasi'] == 'Outlier (Terputus)').sum()
    n_diblokir = (df_r['Kategori_Propagasi'] == '🚫 Dikecualikan (Blacklist Stort)').sum()
    n_sub_clusters = df_r.loc[df_r['Kategori_Propagasi'] == 'Sub-Cluster Valid', 'cluster_id'].nunique()

    print(f"   -> Main Network (Root)                : {n_main} tiang")
    print(f"   -> Sub-Cluster Valid (>= {MIN_TIANG_SUBCLUSTER} tiang) : {n_sub} tiang ({n_sub_clusters} cluster)")
    if n_dekat_padat > 0:
        print(f"   -> Sub-Cluster Valid (Dekat Jaringan Padat) : {n_dekat_padat} tiang "
              f"(diselamatkan dari Outlier, masih <= {JARAK_DEKAT_PADAT_KM} km ke jaringan padat)")
    print(f"   -> Outlier (Terputus / jauh & dikit tiangnya) : {n_outlier} tiang")
    if n_diblokir > 0:
        print(f"   -> 🚫 Dikecualikan (Blacklist Stort)   : {n_diblokir} tiang")
    print()

    info_root_point[reg] = root_point
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
    'Sub-Cluster Valid (Dekat Jaringan Padat)': '#f39c12',
    'Outlier (Terputus)': '#e74c3c',
    '🚫 Dikecualikan (Blacklist Stort)': '#95a5a6',
}
ukuran = {'Main Network (Root)': 15, 'Sub-Cluster Valid': 22,
          'Sub-Cluster Valid (Dekat Jaringan Padat)': 22, 'Outlier (Terputus)': 25,
          '🚫 Dikecualikan (Blacklist Stort)': 18}

for i, reg in enumerate(daftar_regional_proses):
    ax = axes[i // n_cols][i % n_cols]
    df_r = df_all[df_all['regional'] == reg]
    for kat, c in warna.items():
        sub = df_r[df_r['Kategori_Propagasi'] == kat]
        if len(sub) == 0:
            continue
        ax.scatter(sub['lon'], sub['lat'], c=c, s=ukuran[kat], alpha=0.85,
                   edgecolors='black' if kat not in ('Main Network (Root)',) else 'none',
                   linewidths=0.5, label=f'{kat} ({len(sub)})',
                   zorder=2 if kat not in ('Outlier (Terputus)', '🚫 Dikecualikan (Blacklist Stort)') else 3)
    root_point = info_root_point.get(reg)
    if root_point is not None:
        ax.scatter(root_point[0], root_point[1], c='#f1c40f', marker='*', s=350,
                   edgecolors='black', label='Titik Jangkar (Root)', zorder=4)
    ax.set_title(f'{reg}\nRadius: {RADIUS_SCAN_KM} km | Min. Sub-Cluster: {MIN_TIANG_SUBCLUSTER} tiang',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend(loc='upper right', framealpha=0.9, fontsize=8)
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
export_filename = f"TiangJabo_Propagasi_{RADIUS_SCAN_KM}km_min{MIN_TIANG_SUBCLUSTER}_per_regional.csv"
df_export.to_csv(export_filename, index=False)
print(f"✅ File CSV berhasil disimpan: {export_filename}")

# ==========================================
# 6. PETA INTERAKTIF (folium) - basemap OpenStreetMap asli per regional
# ==========================================
if BUAT_PETA_INTERAKTIF:
    try:
        import folium

        MAKS_TITIK_PER_KATEGORI = 4000  # batas titik yg digambar 1-1 per kategori biar peta tetap ringan
        warna_hex = {
            'Main Network (Root)': '#3498db',
            'Sub-Cluster Valid': '#2ecc71',
            'Sub-Cluster Valid (Dekat Jaringan Padat)': '#f39c12',
            'Outlier (Terputus)': '#e74c3c',
            '🚫 Dikecualikan (Blacklist Stort)': '#7f8c8d',
        }

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
            nama_file_peta = f"Peta_Propagasi_{reg.replace(' ', '_')}.html"
            m.save(nama_file_peta)
            print(f"🗺️  Peta interaktif disimpan: {nama_file_peta} (buka di browser buat ngecek visual)")
    except ImportError:
        print("\n⚠️  Package 'folium' belum ke-install, peta interaktif dilewati.")
        print("    Jalankan `!pip install folium -q` di cell terpisah lalu run ulang script ini kalau mau peta-nya.")
