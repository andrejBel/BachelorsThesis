#pragma once
#include <string>
#include <algorithm>
using namespace std;

namespace processing 
{
	class Tokanizer
	{
	public:

		template<typename T>
		struct TokanizerResult
		{
			T result_;
			bool success_ = false;

			operator bool() const 
			{
				return success_;
			}

		};

		Tokanizer(const string& delimiter) :
			delimiter_(delimiter),
			text_(""),
			start_(0),
			end_(0)
		{}

		void setText(const string& text)
		{
			start_ = 0;
			end_ = 0;
			text_ = text;
			if (text_.size() > 0)
			{
				if (text.find(delimiter_) == string::npos)
				{
					text_ += delimiter_;
				}
				else if (text.rfind(delimiter_) != string::npos && text.rfind(delimiter_) != text_.size() - delimiter_.size())
				{
					text_ += delimiter_;
				}
			}
		}

		bool hasNextToken()
		{
			return text_.find(delimiter_, start_) != string::npos;
		}

		string getNextToken()
		{
			end_ = text_.find(delimiter_, start_);
			if (end_ == string::npos)
			{
				throw std::runtime_error("No more tokens!");
			}
			string result(text_.substr(start_, end_ - start_));
			start_ = end_ + delimiter_.size();
			return result;
		}

		TokanizerResult<int> tryGetNextInt()
		{
			int initializer = 0;
			return tryGetNumber([](const string& text) { return std::stoi(text); }, initializer);
		}

		TokanizerResult<float> tryGetNextFloat()
		{
			float initializer = 0.0;
			return tryGetNumber([](const string& text) { return std::stof(text); }, initializer);
		}

	private:
		template <typename T, typename Function>
		TokanizerResult<T> tryGetNumber(Function function, T initial)
		{
			TokanizerResult<T> result;
			size_t end = text_.find(delimiter_, start_);
			if (end == string::npos)
			{
				std::runtime_error("No more tokens!");
			}
			string maybe(text_.substr(start_, end - start_));
			maybe.erase(remove(maybe.begin(), maybe.end(), ' '), maybe.end());
			try
			{
				result.result_ = function(maybe);
				result.success_ = true;
				end_ = end;
				start_ = end_ + delimiter_.size();
			}
			catch (const std::exception&)
			{
				result.result_ = static_cast<T>(initial);
				result.success_ = false;
			}
			return result;
		}

	private:
		string delimiter_;
		string text_;
		size_t start_;
		size_t end_;

	};



}
