# Amazon Product Relevance - category-to-category prototype

Shows which product categories customers explore after the categories they are viewing.
Pick one or several categories, then see a ranked list ("Customers also explore") and a journey tree.

    pip install -r requirements.txt
    python generate_sample_data.py   # only needed to (re)create sample data
    streamlit run app.py

## Data format (data/)
**products.json**: `[{"asin": "B0001", "title": "...", "category": "...", "price": 99}]`

**interactions.json**: `[{"user_id": "U1", "history": [{"asin": "B0001", "rating": 5, "timestamp": "2022-03-01"}]}]`

Each interaction's product is mapped to its category, so the app works at the category level.

## How ranking works
1. For every user, sort their interactions by time and convert each product to its category.
2. For each category, collect the other categories in the user's next `window` interactions.
3. For a selected category: `score = share_of_users x ((1-w) + w x avg_rating/5)`; `relevance = score / sum(scores)`.

## Look-ahead window
Sets how far ahead "next" reaches. A category counts as next if it appears within the customer's next `window` interactions.
For the history Laptop, Mouse, Keyboard, Monitor, the categories after Laptop are:
- window 1: Mouse
- window 2: Mouse, Keyboard
- window 3: Mouse, Keyboard, Monitor

Small windows capture strict "what comes right after" sequences. Larger windows capture "what customers eventually explore" and give higher counts.

## Controls
- **Categories per level**: 1 to 5 buttons.
- **Recommendation levels**: slider, 1 to 3.
- **Look-ahead window**: 1 to 5 buttons (see above).
- **Minimum supporting customers**: Any / 20+ / 50+ / 100+.
- **Rating influence**: Popularity (0%) / Balanced (50%) / Ratings (100%).
- **Summary view**: the corner buttons on each box switch between boxes (top N) and a list (all categories).

## Selecting several categories
- **Show each separately**: one result section per selected category.
- **Combine into one list**: ranks what users explore after viewing *any* of the selected categories (users are counted once, and the selected categories are excluded from the results).

## Share of users vs. relevance
- **Share of users** is raw popularity: what fraction of the selected category's users went to this category. Ratings don't affect it.
- **Relevance** is the app's recommendation strength. It starts from share of users, adjusts it by the average rating, then rescales so all the relevance values in the list add up to 100%.

Example (Laptop has 852 users, Rating influence = 50%):

| Next category | Users who went next | Share of users | Avg rating | Relevance |
|---|---|---|---|---|
| Mouse | 310 | 36% | 4.43 | 18.7% |
| USB-C Hub | 272 | 32% | 4.27 | 16.1% |

At 0% rating influence, the ranking order follows share of users exactly.
