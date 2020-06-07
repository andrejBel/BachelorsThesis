#include "Loader.h"

void Loader::load(const string & filename)
{
	static const vector<string> supportedFormats({ ".jpg", ".png" });
	ifstream inputFile(filename);
	string line;
	Tokanizer tokanizer(DELIMITER);
	vector<float> floats;
	try
	{
		if (inputFile.is_open())
		{
			readLine(inputFile); // convolution type
			insertLineFromStreamIntoTokanizer(inputFile, tokanizer);
			Factory::TypeOfConvolution convolutionType = Factory::fromStringToConvolutionType(tokanizer.getNextToken());
			switch (convolutionType)
			{
			case processing::Factory::TypeOfConvolution::SIMPLE_CONVOLUTION:
			{
				TestBuilder simpleConvolutionBuilder;
				readLine(inputFile); // test type
				insertLineFromStreamIntoTokanizer(inputFile, tokanizer);
				string testTypeString = tokanizer.getNextToken();
				Factory::TestType testType = Factory::fromStringToTestType(testTypeString);
				int numberOfConvoltutionRunnable(-1);
				switch (testType)
				{
				case processing::Factory::TestType::ALONE:
					numberOfConvoltutionRunnable = 1;
					break;
				case processing::Factory::TestType::AGAINST:
					numberOfConvoltutionRunnable = 2;
					break;
				case processing::Factory::TestType::NONE:
					throw std::runtime_error("Unknown test type");
					break;
				default:
					break;
				}
				readLine(inputFile); // kernels aka runnables
				insertLineFromStreamIntoTokanizer(inputFile, tokanizer); // our runnable
				for (int runnableIndex = 0; runnableIndex < numberOfConvoltutionRunnable; runnableIndex++)
				{
					if (tokanizer.hasNextToken())
					{
						string runnableName = tokanizer.getNextToken();
						Factory::SimpleConvolution runnableType = Factory::fromStringToSimpleRunnableType(runnableName);
						if (runnableType != Factory::SimpleConvolution::NONE)
						{
							simpleConvolutionBuilder.addRunnable(Factory::getSimpleConvolutionFactory().getRunnable(runnableType));
						}
						else
						{
							throw std::runtime_error("Unknown runnable type");
						}
					}
					else
					{
						throw::std::runtime_error("Unable to read runnable type");
					}
				}
				readLine(inputFile); // epsilon
				insertLineFromStreamIntoTokanizer(inputFile, tokanizer);
				auto epsilon = tokanizer.tryGetNextFloat();
				if (epsilon)
				{
					simpleConvolutionBuilder.setEpsilon(epsilon.result_);
				}
				else
				{
					throw std::runtime_error("Unable to read epsilon");
				}
				readLine(inputFile); // output
				insertLineFromStreamIntoTokanizer(inputFile, tokanizer);
				string stringOutputType = tokanizer.getNextToken();
				Factory::OutputType outputType =  Factory::fromStringToOutputType(stringOutputType);
				simpleConvolutionBuilder.setOutputType(outputType);
				if (outputType == Factory::OutputType::TEXTFILE) 
				{
					simpleConvolutionBuilder.setOutputPath(tokanizer.getNextToken());
				}
				readLine(inputFile); // number of images
				insertLineFromStreamIntoTokanizer(inputFile, tokanizer);
				auto numberOfPictures = tokanizer.tryGetNextInt();
				if (numberOfPictures)
				{
					readLine(inputFile); // image paths
					for (int i = 0; i < numberOfPictures.result_; i++)
					{
						insertLineFromStreamIntoTokanizer(inputFile, tokanizer);
						string filePath = tokanizer.getNextToken();
						bool isFormatSupported = false;
						for (auto& extension : supportedFormats)
						{
							if (filePath.find(extension) != string::npos)
							{
								isFormatSupported = true;
								break;
							}
						}
						if (isFormatSupported)
						{
							simpleConvolutionBuilder.addImagePath(filePath);
						}
						else
						{
							throw std::runtime_error("Format not supported");
						}
					}
				}
				else
				{
					throw std::runtime_error("Unable to read number of images");
				}
				readLine(inputFile); // number of filters
				insertLineFromStreamIntoTokanizer(inputFile, tokanizer);
				auto numberOfFilters = tokanizer.tryGetNextInt();
				if (numberOfFilters)
				{
					for (int filterIndex = 0; filterIndex < numberOfFilters.result_; filterIndex++)
					{
						readLine(inputFile); // FILTERS WIDTH | FILTER MULTIPLIER
						insertLineFromStreamIntoTokanizer(inputFile, tokanizer);
						auto filterWidth = tokanizer.tryGetNextInt();
						auto multiplier = tokanizer.tryGetNextFloat();
						if (filterWidth)
						{
							Tokanizer::TokanizerResult<float> filterValue;
							vector<float> filterValues;
							if (multiplier)
							{
								readLine(inputFile); // FILTER values
								for (int filterLine = 0; filterLine < filterWidth.result_; ++filterLine)
								{
									insertLineFromStreamIntoTokanizer(inputFile, tokanizer);
									for (int index = 0; index < filterWidth.result_; ++index)
									{
										filterValue = tokanizer.tryGetNextFloat();
										if (filterValue)
										{
											filterValues.push_back(filterValue.result_);
										}
										else
										{
											throw std::runtime_error("Unable to read filter value");
										}
									}
								}
								simpleConvolutionBuilder.addFilter(createFilter(filterWidth.result_, filterValues, multiplier.result_));

							}
							else
							{
								throw std::runtime_error("Unable to read filter multiplier");
							}
						}
						else
						{
							throw std::runtime_error("Unable to read filter width");
						}
					}
				}
				else
				{
					throw std::runtime_error("Unable to read number of filters");
				}
				simpleConvolutionBuilder.build().run();
				break;
			}
			case processing::Factory::TypeOfConvolution::MULTI_CONVOLUTION:
			{
				TestMultiBuilder multiConvolutionBuilder;
				readLine(inputFile); // test type
				insertLineFromStreamIntoTokanizer(inputFile, tokanizer);
				string testTypeString = tokanizer.getNextToken();
				Factory::TestType testType = Factory::fromStringToTestType(testTypeString);
				int numberOfConvoltutionRunnable(-1);
				switch (testType)
				{
				case processing::Factory::TestType::ALONE:
					numberOfConvoltutionRunnable = 1;
					break;
				case processing::Factory::TestType::AGAINST:
					numberOfConvoltutionRunnable = 2;
					break;
				case processing::Factory::TestType::NONE:
					throw std::runtime_error("Unknown test type");
					break;
				default:
					break;
				}
				readLine(inputFile); // runnables
				insertLineFromStreamIntoTokanizer(inputFile, tokanizer); // our runnable
				for (int runnableIndex = 0; runnableIndex < numberOfConvoltutionRunnable; runnableIndex++)
				{
					if (tokanizer.hasNextToken())
					{
						string runnableName = tokanizer.getNextToken();
						Factory::MultiConvolution runnableType = Factory::fromStringToMultiRunnableType(runnableName);
						if (runnableType != Factory::MultiConvolution::NONE)
						{
							multiConvolutionBuilder.addRunnable(Factory::getMultiConvolutionFactory().getRunnable(runnableType));
						}
						else
						{
							throw std::runtime_error("Unknown runnable type");
						}
					}
					else
					{
						throw::std::runtime_error("Unable to read runnable type");
					}
				}
				readLine(inputFile); // epsilon
				insertLineFromStreamIntoTokanizer(inputFile, tokanizer);
				auto epsilon = tokanizer.tryGetNextFloat();
				if (epsilon)
				{
					multiConvolutionBuilder.setEpsilon(epsilon.result_);
				}
				else
				{
					throw std::runtime_error("Unable to read epsilon");
				}
				readLine(inputFile); // output
				insertLineFromStreamIntoTokanizer(inputFile, tokanizer);
				string stringOutputType = tokanizer.getNextToken();
				Factory::OutputType outputType = Factory::fromStringToOutputType(stringOutputType);
				multiConvolutionBuilder.setOutputType(outputType);
				if (outputType == Factory::OutputType::TEXTFILE)
				{
					multiConvolutionBuilder.setOutputPath(tokanizer.getNextToken());
				}
				readLine(inputFile); // number of pictures
				insertLineFromStreamIntoTokanizer(inputFile, tokanizer);
				auto numberOfPictures = tokanizer.tryGetNextInt();
				if (numberOfPictures)
				{
					readLine(inputFile); // image paths
					for (int i = 0; i < numberOfPictures.result_; i++)
					{
						insertLineFromStreamIntoTokanizer(inputFile, tokanizer);
						string filePath = tokanizer.getNextToken();
						bool isFormatSupported = false;
						for (auto& extension : supportedFormats)
						{
							if (filePath.find(extension) != string::npos)
							{
								isFormatSupported = true;
								break;
							}
						}
						if (isFormatSupported)
						{
							multiConvolutionBuilder.addImagePath(filePath);
						}
						else
						{
							throw std::runtime_error("Format not supported");
						}
					}
				}
				else
				{
					throw std::runtime_error("Unable to read number of images");
				}
				readLine(inputFile); // filter group size
				insertLineFromStreamIntoTokanizer(inputFile, tokanizer);
				auto numberOfFilterGroups = tokanizer.tryGetNextInt();
				if (numberOfFilterGroups)
				{
					for (int filterGroupIndex = 0; filterGroupIndex < numberOfFilterGroups.result_; filterGroupIndex++)
					{
						readLine(inputFile); // line to separate filter groups
						vector<shared_ptr<Filter>> filterGroup;
						readLine(inputFile); // filter width
						insertLineFromStreamIntoTokanizer(inputFile, tokanizer);
						auto filterWidth = tokanizer.tryGetNextInt();
						if (filterWidth)
						{
							for (int filterIndex = 0; filterIndex < numberOfPictures.result_; filterIndex++)
							{
								readLine(inputFile); // multiplier
								insertLineFromStreamIntoTokanizer(inputFile, tokanizer);
								auto multiplier = tokanizer.tryGetNextFloat();
								Tokanizer::TokanizerResult<float> filterValue;
								vector<float> filterValues;
								if (multiplier)
								{
									readLine(inputFile); // filter values
									for (int filterLine = 0; filterLine < filterWidth.result_; ++filterLine)
									{
										insertLineFromStreamIntoTokanizer(inputFile, tokanizer);
										for (int index = 0; index < filterWidth.result_; ++index)
										{
											filterValue = tokanizer.tryGetNextFloat();
											if (filterValue)
											{
												filterValues.push_back(filterValue.result_);
											}
											else
											{
												throw std::runtime_error("Unable to read filter value");
											}
										}
									}
									filterGroup.push_back(createFilter(filterWidth.result_, filterValues, multiplier.result_));
								}
								else
								{
									throw std::runtime_error("Unable to read filter multiplier");
								}
							}
						}
						else
						{
							throw std::runtime_error("Unable to read filter width");
						}
						multiConvolutionBuilder.addFilterGroup(filterGroup);
					}
				}
				else
				{
					throw std::runtime_error("Unable to read number of filters");
				}
				multiConvolutionBuilder.build().run();
				break;
			}
			case processing::Factory::TypeOfConvolution::NONE:
				throw std::runtime_error("Unknown convolution type");
				break;
			default:
				break;
			}

		}
		else
		{
			std::cerr << "File was not found" << endl;
			return;
		}
	}
	catch (const std::runtime_error& error)
	{
		std::cerr << "Error: " << error.what() << endl;
		return;
	}
	catch (const std::exception& error)
	{
		std::cerr << "Error reading file. Error: " << error.what() << endl;
		return;
	}
}
