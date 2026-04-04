#include <vector>
#include <utility>

std::pair<std::vector<int>, long long> transaction(long long M);   // provided by the judge

void buy_souvenirs(int N, long long P0)
{
    long long M1 = P0 - 1;
    auto [L1, R1] = transaction(M1);

    if (L1.size() == 2) {
        long long S  = M1 - R1;
        long long M2 = S / 2;
        transaction(M2);
        return;
    }

    long long P1 = M1 - R1;
    long long M  = P1 - 1;
    transaction(M);
    transaction(M);
}
