import random


# 1. Konfigurasi 
TARGET = "ZAINUL MUTAWAKKIL"
GENES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ "
POP_SIZE = 100
def get_fitness(guess):
    """Menghitung skor: berapa banyak huruf yang cocok dengan target"""
    score = 0
    for i in range(len(TARGET)):
        if guess[i] == TARGET[i]:
            score += 1
    return score
def mutate(parent):
    """Mengubah satu huruf secara acak (Mutasi)"""
    index = random.randrange(0, len(parent))
    child_genes = list(parent)
    child_genes[index] = random.choice(GENES)
    return "".join(child_genes)

#Inisialisasi: Buat populasi awal secara acak
population = []
for _ in range(POP_SIZE):
    indiv = "".join(random.choice(GENES) for _ in range(len(TARGET)))
    population.append(indiv)
generation = 1
while True:
    # 2. Evaluasi Fitness & Sortir (Seleksi sederhana: ambil yang terbaik)
    population = sorted(population, key=lambda x: get_fitness(x), reverse=True)
    best_fitness = get_fitness(population[0])
    print(f"Generasi {generation}: {population[0]} (Fitness: {best_fitness})")
    # Jika target tercapai, berhenti
    if best_fitness >= len(TARGET):
        break
    # 3. Reproduksi (Membuat generasi baru)
    new_generation = population[:10] # Elitism: Ambil 10 terbaik tanpa perubahan
    while len(new_generation) < POP_SIZE:
        parent = random.choice(population[:50]) # Seleksi: Pilih dari 50 besar
        child = mutate(parent) 
        new_generation.append(child)
    population = new_generation
    generation += 1