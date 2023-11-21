#include <iostream>
#include <vector>
#include <string>

using namespace std;

class Solution {
  public:
      int evalRPN(vector<string>& tokens) {
          int val = 0;
          for (int i=0;i<tokens.size();++i) {
              cout << "Checking at " << i << endl;
              switch (tokens[i][0]) {
                  case '*':
                      tokens[i-2] = to_string(stoi(tokens[i-2])*stoi(tokens[i-1]));
                      tokens.erase(tokens.begin()+i);
                      tokens.erase(tokens.begin()+i-2);
                      --i;
                  case '/':
                      tokens[i-2] = to_string(stoi(tokens[i-2])/stoi(tokens[i-1]));
                      tokens.erase(tokens.begin()+i);
                      tokens.erase(tokens.begin()+i-2);
                      --i;
                  case '+':
                      cout << "+" << endl;
                      tokens[i-2] = to_string(stoi(tokens[i-2])+stoi(tokens[i-1]));
                      tokens.erase(tokens.begin()+i);
                      cout << to_string(stoi(tokens[i-2])) << endl;
                      tokens.erase(tokens.begin()+i-2);
                      cout << i << endl;
                      --i;
                  case '-':
                      tokens[i-2] = to_string(stoi(tokens[i-2])-stoi(tokens[i-1]));
                      tokens.erase(tokens.begin()+i);
                      tokens.erase(tokens.begin()+i-2);
                      --i;
              }
          }
          return 1;
      }
};

int main() {
    Solution solution;
    vector<string> input = {"2","1","+","3","*"};
    cout << solution.evalRPN(input);
}