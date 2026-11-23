"""Generate sample products.json and interactions.json for the prototype.

Your teammate can replace the two files in data/ with real Amazon data as long
as the same fields are kept (see README.md).
"""
import json
import random
from datetime import date, timedelta
from pathlib import Path

random.seed(42)
DATA_DIR = Path(__file__).parent / "data"

# category -> [(title, price, typical rating)]
CATALOG = {
    "Laptop": [("Dell XPS 13 Laptop", 999, 4.5), ("Apple MacBook Air M2", 1099, 4.7),
               ("Lenovo ThinkPad E14", 749, 4.2), ("HP Pavilion 15 Laptop", 619, 3.9)],
    "Mouse": [("Logitech MX Master 3S", 99, 4.7), ("Logitech M185 Wireless Mouse", 15, 4.4),
              ("Razer DeathAdder V3", 69, 4.5), ("Anker Vertical Ergonomic Mouse", 20, 4.1)],
    "Keyboard": [("Keychron K2 Mechanical Keyboard", 89, 4.6), ("Logitech K380 Bluetooth Keyboard", 40, 4.5),
                 ("Apple Magic Keyboard", 99, 4.3), ("Redragon K552 Keyboard", 36, 4.4)],
    "Monitor": [("Dell UltraSharp 27 Monitor", 429, 4.6), ("LG 27 4K UHD Monitor", 299, 4.4),
                ("Samsung 24 FHD Monitor", 139, 4.3), ("ASUS ProArt 32 Monitor", 549, 4.5)],
    "Headphones": [("Sony WH-1000XM5", 349, 4.6), ("Bose QuietComfort 45", 279, 4.5),
                   ("Anker Soundcore Q30", 79, 4.4), ("JBL Tune 510BT", 39, 4.2)],
    "Webcam": [("Logitech C920 HD Webcam", 69, 4.5), ("Logitech Brio 4K Webcam", 169, 4.3),
               ("Razer Kiyo Webcam", 79, 4.0), ("Anker PowerConf C200", 49, 4.4)],
    "USB-C Hub": [("Anker 7-in-1 USB-C Hub", 36, 4.5), ("Satechi Multiport Adapter", 79, 4.4),
                  ("UGREEN 5-in-1 Hub", 25, 4.3), ("Dell Thunderbolt Dock", 229, 4.0)],
    "Laptop Stand": [("Rain Design mStand", 45, 4.6), ("Nulaxy Adjustable Stand", 26, 4.5),
                     ("Roost Portable Stand", 75, 4.4), ("Twelve South Curve", 55, 4.6)],
    "External SSD": [("Samsung T7 Portable SSD", 89, 4.7), ("SanDisk Extreme Portable SSD", 99, 4.5),
                     ("Crucial X6 SSD", 79, 4.5), ("WD My Passport SSD", 109, 4.3)],
    "Backpack": [("Peak Design Everyday Backpack", 279, 4.6), ("Matein Travel Laptop Backpack", 36, 4.6),
                 ("Samsonite Classic Backpack", 59, 4.3), ("Targus Business Backpack", 45, 4.2)],
}

# category -> relative likelihood of the next category
AFFINITY = {
    "Laptop": {"Mouse": 22, "Backpack": 18, "USB-C Hub": 18, "Laptop Stand": 12, "External SSD": 12, "Keyboard": 8, "Monitor": 10},
    "Mouse": {"Keyboard": 30, "Laptop Stand": 15, "Monitor": 15, "Headphones": 10, "USB-C Hub": 10, "Webcam": 10, "Backpack": 10},
    "Keyboard": {"Mouse": 30, "Monitor": 20, "Headphones": 15, "Laptop Stand": 15, "Webcam": 10, "USB-C Hub": 10},
    "Monitor": {"USB-C Hub": 25, "Webcam": 20, "Keyboard": 15, "Mouse": 15, "Headphones": 15, "Laptop Stand": 10},
    "Headphones": {"Webcam": 25, "Mouse": 15, "Keyboard": 15, "Monitor": 15, "External SSD": 10, "Laptop": 10, "USB-C Hub": 10},
    "Webcam": {"Headphones": 30, "Monitor": 20, "Laptop Stand": 15, "Keyboard": 15, "Mouse": 10, "USB-C Hub": 10},
    "USB-C Hub": {"Monitor": 25, "External SSD": 25, "Mouse": 15, "Keyboard": 15, "Laptop Stand": 10, "Webcam": 10},
    "Laptop Stand": {"Keyboard": 25, "Mouse": 25, "Monitor": 20, "USB-C Hub": 15, "Webcam": 15},
    "External SSD": {"Backpack": 20, "USB-C Hub": 20, "Laptop": 20, "Laptop Stand": 10, "Mouse": 15, "Keyboard": 15},
    "Backpack": {"Laptop Stand": 20, "External SSD": 20, "Mouse": 20, "Headphones": 20, "USB-C Hub": 20},
}


def main(n_users: int = 2000) -> None:
    products, by_category, popularity, quality = [], {}, {}, {}
    n = 1
    for category, items in CATALOG.items():
        for title, price, rating in items:
            asin = f"B{n:04d}"
            n += 1
            products.append({"asin": asin, "title": title, "category": category, "price": price})
            by_category.setdefault(category, []).append(asin)
            popularity[asin] = random.uniform(1, 4)
            quality[asin] = rating

    def pick(category, seen):
        options = [a for a in by_category[category] if a not in seen]
        if not options:
            return None
        return random.choices(options, weights=[popularity[a] for a in options])[0]

    users = []
    for u in range(1, n_users + 1):
        category = random.choices(list(CATALOG), weights=[5 if c == "Laptop" else 1 for c in CATALOG])[0]
        seen, history = set(), []
        when = date(2022, 1, 1) + timedelta(days=random.randint(0, 400))
        for _ in range(random.randint(3, 8)):
            asin = pick(category, seen)
            if asin is None:
                break
            seen.add(asin)
            rating = min(5, max(1, round(random.gauss(quality[asin], 0.8))))
            history.append({"asin": asin, "rating": rating, "timestamp": when.isoformat()})
            when += timedelta(days=random.randint(1, 30))
            nxt = AFFINITY[category]
            category = random.choices(list(nxt), weights=list(nxt.values()))[0]
        users.append({"user_id": f"U{u:05d}", "history": history})

    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "products.json").write_text(json.dumps(products, indent=2))
    (DATA_DIR / "interactions.json").write_text(json.dumps(users))
    print(f"Wrote {len(products)} products and {len(users)} users to {DATA_DIR}")


if __name__ == "__main__":
    main()
