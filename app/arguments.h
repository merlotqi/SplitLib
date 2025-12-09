#pragma once

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

class AValue {
 public:
  std::vector<std::string> values;

  AValue(std::vector<std::string> values) { this->values = values; }

  template <typename T>
  T as(T alternative) {
    return !values.empty() ? T(values[0]) : alternative;
  }

  template <typename T>
  T as() {
    return !values.empty() ? T(values[0]) : T();
  }
};

template <>
std::vector<std::string> AValue::as<std::vector<std::string>>(
    std::vector<std::string> alternative) {
  return !values.empty() ? values : alternative;
}

template <>
std::vector<std::string> AValue::as<std::vector<std::string>>() {
  return !values.empty() ? values : std::vector<std::string>{};
}

template <>
std::vector<double> AValue::as<std::vector<double>>(
    std::vector<double> alternative) {
  std::vector<double> res;
  for (auto& v : values) {
    res.push_back(std::stod(v));
  }
  return !res.empty() ? res : alternative;
}

template <>
std::vector<double> AValue::as<std::vector<double>>() {
  return as<std::vector<double>>({});
}

template <>
double AValue::as<double>(double alternative) {
  return !values.empty() ? std::stod(values[0]) : alternative;
}

template <>
double AValue::as<double>() {
  return !values.empty() ? std::stod(values[0]) : 0.0;
}

template <>
int AValue::as<int>(int alternative) {
  return !values.empty() ? std::stoi(values[0]) : alternative;
}

template <>
int AValue::as<int>() {
  return !values.empty() ? std::stoi(values[0]) : 0;
}

class Argument {
 private:
  std::vector<std::string> split(std::string str,
                                 std::vector<char> delimiters) {
    std::vector<std::string> tokens;

    auto isDelimiter = [&delimiters](char ch) {
      for (auto& delimiter : delimiters) {
        if (ch == delimiter) {
          return true;
        }
      }

      return false;
    };

    int start = 0;
    for (int i = 0; i < str.size(); i++) {
      if (isDelimiter(str[i])) {
        if (start < i) {
          auto token = str.substr(start, i - start);
          tokens.push_back(token);
        } else {
          tokens.push_back("");
        }

        start = i + 1;
      }
    }

    if (start < str.size()) {
      tokens.push_back(str.substr(start));
    } else if (isDelimiter(str[str.size() - 1])) {
      tokens.push_back("");
    }

    return tokens;
  }

 public:
  std::string id = "";
  std::string description = "";

  Argument(std::string id, std::string description) {
    this->id = id;
    this->description = description;
  }

  bool is(std::string name) {
    auto tokens = split(id, {','});

    for (auto token : tokens) {
      if (token == name) {
        return true;
      }
    }

    return false;
  }

  std::string fullname() {
    auto tokens = split(id, {','});

    for (auto token : tokens) {
      if (token.size() > 1) {
        return token;
      }
    }

    return "";
  }

  std::string shortname() {
    auto tokens = split(id, {','});

    for (auto token : tokens) {
      if (token.size() == 1) {
        return token;
      }
    }

    return "";
  }
};

class Arguments {
  bool startsWith(const std::string& str, const std::string& prefix) {
    if (str.size() < prefix.size()) {
      return false;
    }

    return str.substr(0, prefix.size()).compare(prefix) == 0;
  }

 public:
  int argc = 0;
  char** argv = nullptr;

  bool ignoreFirst = true;

  std::vector<std::string> tokens;
  std::vector<Argument> argdefs;
  std::unordered_map<std::string, std::vector<std::string>> map;

  Arguments(int argc, char** argv, bool ignoreFirst = true) {
    this->argc = argc;
    this->argv = argv;
    this->ignoreFirst = ignoreFirst;

    for (int i = ignoreFirst ? 1 : 0; i < argc; i++) {
      std::string token = std::string(argv[i]);
      tokens.push_back(token);
    }

    std::string currentKey = "";
    map.insert({currentKey, {}});
    for (std::string token : tokens) {
      if (startsWith(token, "---")) {
        std::cerr << "Invalid argument: " << token << std::endl;
        exit(1);
      } else if (startsWith(token, "--")) {
        currentKey = token.substr(2);
        map.insert({currentKey, {}});
      } else if (startsWith(token, "-")) {
        currentKey = token.substr(1);
        map.insert({currentKey, {}});
      } else {
        map[currentKey].push_back(token);
      }
    }
  }

  void addArgument(std::string id, std::string description) {
    Argument arg(id, description);

    argdefs.push_back(arg);
  }

  Argument* getArgument(std::string name) {
    for (Argument& arg : argdefs) {
      if (arg.is(name)) {
        return &arg;
      }
    }

    return nullptr;
  }

  std::vector<std::string> keys() {
    std::vector<std::string> keys;
    for (auto entry : map) {
      keys.push_back(entry.first);
    }

    return keys;
  }

  std::string usage() {
    std::stringstream ss;

    std::vector<std::string> keys;

    for (auto argdef : argdefs) {
      std::stringstream ssKey;
      if (!argdef.shortname().empty()) {
        ssKey << "  -" << argdef.shortname();

        if (!argdef.fullname().empty()) {
          ssKey << " [ --" << argdef.fullname() << " ]";
        }

      } else if (!argdef.fullname().empty()) {
        ssKey << "  --" << argdef.fullname();
      }

      keys.push_back(ssKey.str());
    }

    int keyColumnLength = 0;
    for (auto key : keys) {
      keyColumnLength = std::max(int(key.size()), keyColumnLength);
    }
    keyColumnLength = keyColumnLength + 2;

    for (int i = 0; i < argdefs.size(); i++) {
      keys[i].resize(keyColumnLength, ' ');
      ss << keys[i] << argdefs[i].description << std::endl;
    }

    return ss.str();
  }

  bool has(std::string name) {
    Argument* arg = getArgument(name);

    if (arg == nullptr) {
      return false;
    }

    for (auto entry : map) {
      if (arg->is(entry.first)) {
        return true;
      }
    }

    return false;
  }

  AValue get(std::string name) {
    Argument* arg = getArgument(name);

    std::vector<std::string> values;

    for (auto entry : map) {
      if (arg->is(entry.first)) {
        values.insert(values.end(), entry.second.begin(), entry.second.end());
        // return AValue(entry.second);
      }
    }

    return AValue(values);
  }
};