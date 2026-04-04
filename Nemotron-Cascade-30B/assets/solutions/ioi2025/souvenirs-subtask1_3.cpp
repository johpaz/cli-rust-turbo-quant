#include <vector>
#include <utility>

using namespace std;

// Provided by the judge.  It returns the list of types bought and the
// number of coins that remain after giving M coins.
pair<vector<int>, long long> transaction(long long M);

void buy_souvenirs(int N, long long P0)
{
    vector<long long> price(N, 0);
    vector<int>      bought(N, 0);

    price[0] = P0;                     // price of type 0 is known

    // -------- Phase 1 : discover all prices ----------
    for (int i = 1; i < N; ++i) {
        long long M = price[i - 1] - 1;                // guaranteed < price[i-1] and ≥ price[i]
        auto [L, R] = transaction(M);                  // L – bought types, R – remaining coins

        for (int t : L) ++bought[t];

        long long spent = M - R;                       // total money spent in this transaction

        if (L.size() == 1) {
            price[i] = spent;                          // only type i was bought
        } else {
            price[i] = spent - 1;                       // cheapest type (price 1) was also bought
            int extra = -1;
            for (int t : L) if (t != i) { extra = t; break; }
            if (extra != -1 && price[extra] == 0) {
                price[extra] = 1;                       // cheapest type has price 1
            }
        }
    }

    // -------- Phase 2 : buy the remaining copies ----------
    for (int i = 1; i < N; ++i) {
        while (bought[i] < i) {
            long long M = price[i];                    // give exactly the price of type i
            auto [L, R] = transaction(M);
            for (int t : L) ++bought[t];
        }
    }
}
