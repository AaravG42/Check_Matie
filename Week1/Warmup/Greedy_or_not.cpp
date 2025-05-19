#include <iostream>
#include <vector>

using namespace std;

int player1score(const vector<int>& list, bool turn, int l, int r) {
    if (l>=r) {
        return turn ? list[l] : 0;
    }
    if (turn) {
        return max(list[l]+player1score(list, !turn, l+1, r), list[r]+player1score(list, !turn, l, r-1));
    } else {
        return min(player1score(list, !turn, l+1, r), player1score(list, !turn, l, r-1));
    }
}

int main() {
    vector<int> list;
    int n; cin>>n;
    int sum = 0;
    for (int i=0; i<n; ++i) {
        int num; cin>>num;
        list.push_back(num);
        sum+=num;
    }
    int p1score = player1score(list, true, 0, list.size()-1);
    if (p1score>(sum-p1score)) cout << "Player 1 wins" << endl;
    else if (p1score<(sum-p1score)) cout << "Player 2 wins" << endl;
    else cout << "Its a draw" << endl;
}