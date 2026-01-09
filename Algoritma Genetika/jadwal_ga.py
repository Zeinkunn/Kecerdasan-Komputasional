import random
import time

# --- Konfigurasi Masalah ---
# Mata Kuliah: A, B, C
# Durasi: 2 Jam masing-masing
# Constraint 1: Tidak boleh overlap
# Constraint 2: A (Prof X) harus SELESAI sebelum jam 10.00
#               (Asumsi jam operasional mulai jam 07:00 atau 08:00)

COURSES = ['A', 'B', 'C']
DURATIONS = {'A': 2, 'B': 2, 'C': 2}

# Range jam yang mungkin (misal jam 8 pagi sampai jam 16 sore last start)
START_HOUR_MIN = 8
START_HOUR_MAX = 16 

POPULATION_SIZE = 10
GENERATIONS = 50
MUTATION_RATE = 0.2

# --- Representasi Kromosom ---
# List of start times [start_A, start_B, start_C]
# start_time: integer jam (misal 8 berarti 08:00)

def create_individual():
    """Membuat satu individu acak."""
    return [random.randint(START_HOUR_MIN, START_HOUR_MAX) for _ in COURSES]

def fitness(individual):
    """
    Menghitung skor fitness. Semakin tinggi semakin baik.
    Penalty diberikan jika melanggar constraint.
    """
    start_A, start_B, start_C = individual
    starts = {'A': start_A, 'B': start_B, 'C': start_C}
    
    penalty = 0
    
    # 1. Cek Overlap
    # Untuk setiap pasang mata kuliah, cek apakah waktunya beririsan
    # Irisan terjadi jika: start1 < end2 AND start2 < end1
    course_list = list(starts.keys())
    for i in range(len(course_list)):
        for j in range(i + 1, len(course_list)):
            c1 = course_list[i]
            c2 = course_list[j]
            
            s1 = starts[c1]
            e1 = s1 + DURATIONS[c1]
            s2 = starts[c2]
            e2 = s2 + DURATIONS[c2]
            
            # Cek overlap
            if s1 < e2 and s2 < e1:
                penalty += 10 # Penalty besar untuk overlap
                
    # 2. Cek Batasan Prof X (Course A)
    # Harus selesai sebelum 10.00
    end_A = starts['A'] + DURATIONS['A']
    if end_A > 10:
        penalty += 10 # Penalty besar jika lewat jam 10
        # Tambahan penalty proporsional seberapa `telat`
        penalty += (end_A - 10) * 1

    # Fitness = 1 / (1 + penalty). 
    # Jika penalty 0 (solusi valid), fitness = 1.
    return 1.0 / (1.0 + penalty)

def selection(population):
    """Tournament selection"""
    k = 3
    tournament = random.sample(population, k)
    best = max(tournament, key=fitness)
    return best

def crossover(p1, p2):
    """Single point crossover"""
    point = random.randint(1, len(p1)-1)
    c1 = p1[:point] + p2[point:]
    c2 = p2[:point] + p1[point:]
    return c1, c2

def mutate(individual):
    """Mengubah jam mulai salah satu course secara acak"""
    if random.random() < MUTATION_RATE:
        idx = random.randint(0, len(individual)-1)
        # Ubah jam ke nilai baru
        individual[idx] = random.randint(START_HOUR_MIN, START_HOUR_MAX)
    return individual

def run_ga():
    random.seed(42) # Agar reproducible
    
    # 1. Inisialisasi Populasi
    population = [create_individual() for _ in range(POPULATION_SIZE)]
    
    print(f"Mulai Evolusi dengan {POPULATION_SIZE} individu selama {GENERATIONS} generasi...")
    
    best_solution = None
    best_fit = -1
    
    for gen in range(GENERATIONS):
        # Hitung fitness
        current_best = max(population, key=fitness)
        current_fit = fitness(current_best)
        
        if current_fit > best_fit:
            best_fit = current_fit
            best_solution = current_best
        
        # Print progres setiap generasi
        print(f"Generasi {gen}: Fitness Terbaik = {best_fit:.4f}, Jadwal = {best_solution}")
        
        # Jika menemukan solusi sempurna (fitness = 1.0), berhenti
        if best_fit >= 1.0:
            print(f"Solusi optimal ditemukan pada generasi {gen}!")
            break
            
        # Buat generasi baru
        new_pop = []
        while len(new_pop) < POPULATION_SIZE:
            p1 = selection(population)
            p2 = selection(population)
            c1, c2 = crossover(p1, p2)
            new_pop.append(mutate(c1))
            if len(new_pop) < POPULATION_SIZE:
                new_pop.append(mutate(c2))
        
        population = new_pop
        
    # Output Hasil
    print("\nHasil Terbaik:")
    print(f"Jadwal (Start Times): {best_solution}")
    print(f"Jadwal Detail:")
    names = {'A': 'Kecerdasan Komputasional (Prof X)', 
             'B': 'Struktur Data (Dr Y)', 
             'C': 'Sistem Operasi (Prof Z)'}
    
    # Sort berdasarkan jam mulai
    schedule = []
    keys = ['A', 'B', 'C']
    for k in keys:
        schedule.append((k, best_solution[keys.index(k)]))
    
    schedule.sort(key=lambda x: x[1])
    
    for code, start in schedule:
        end = start + DURATIONS[code]
        print(f"- {names[code]}: {start:02d}.00 - {end:02d}.00")
        
    # Validasi Constraint Check
    if best_fit >= 1.0:
        print("\nSTATUS: VALID (Semua aturan terpenuhi)")
    else:
        print("\nSTATUS: TIDAK VALID (Masih ada bentrok atau pelanggaran)")

if __name__ == "__main__":
    run_ga()
