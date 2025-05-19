#include <iostream>
#include <vector>

using namespace std;

vector<vector<vector<int>>> memo;

int player1score_dp(const vector<int>& list, bool turn, int l, int r) {
    if (l >= r) {
        return turn ? list[l] : 0;
    }
    int turn_idx = turn ? 1 : 0; 
    if (memo[l][r][turn_idx] != -1) {
        return memo[l][r][turn_idx];
    }
    int result;
     if (turn) {
        result =  max(list[l]+player1score_dp(list, !turn, l+1, r), list[r]+player1score_dp(list, !turn, l, r-1));
    } else {
        result = min(player1score_dp(list, !turn, l+1, r), player1score_dp(list, !turn, l, r-1));
    }
    return memo[l][r][turn_idx] = result;
}

int main() {
    vector<int> list;
    int n; cin >> n;
    int sum = 0;
    for (int i = 0; i < n; ++i) {
        int num; cin >> num;
        list.push_back(num);
        sum += num;
    }
    memo.assign(n, vector<vector<int>>(n, vector<int>(2, -1)));
    int p1score = player1score_dp(list, true, 0, list.size() - 1);
    if (p1score > (sum - p1score)) cout << "Player 1 wins" << endl;
    else if (p1score < (sum - p1score)) cout << "Player 2 wins" << endl;
    else cout << "Its a draw" << endl;
}