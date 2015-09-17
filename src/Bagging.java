/*************************************************************************************************************************************************************
 * @author Akshay
 * 
 *         A program which bags the training data and each bag is trained and classified using ID3 algorithm to build decision tree
 * 
 *         Dependencies : TrainData.java, TreeData.java
 *         Compilation : javac directory_path\*.java
 * 
 *         Input : file path of train and test data and number of bags to be created as command line arguments
 *         Output : Decision tree identified using train data and accuracy of decision tree for train and test data
 *         Execution: java Bagging "path_of_train_file" "path_of_test_file" Number of iterations
 *
 *         Example
 *         C:\> java Bagging "D:\Machine Learning\Homework\HW1\Data\train.dat" "D:\Machine Learning\Homework\HW1\Data\test.dat" 50
 *         Accuracy of test file with train data ( 216 instances ) = 82.87
 *         Accuracy of test file with bag data ( 216 instances, 1000 bags ) = 97.22
 ****************************************************************************************************************************************************/

/**
 * This class takes train file as an argument and process it to build a bags of decision trees, then these trees are tested for accuracy test file.
 */


import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Random;

public class Bagging
{
	public static void main( String[] args )
	{
		try
		{
			if ( args.length < 3 )
			{
				System.out.println( "Please input train file, test file and number of iterations as command line arguments." );
				return;
			}

			int k = Integer.parseInt( args[ 2 ] );
			ArrayList< TreeData > baggedList = new ArrayList< TreeData >( k );
			TrainData train_data = new TrainData( args[ 0 ] ); // This is a custom class which stores the train data from file.
			TreeData treeData = train_data.buildTree();

			for ( int i = 0; i < k; i++ )
			{
				TrainData baggedData = getBootstrappedData( train_data );
				baggedList.add( baggedData.buildTree() );
			}

			train_data.calculateAccuracyOfTestFile( treeData, args[ 1 ] );
			train_data.calculateAccuracyOfTestFile( baggedList, args[ 1 ] );
		}
		catch ( FileNotFoundException e )
		{
			// Do nothing already output is printed
		}
	}

	/**
	 * Creates a copy of train data using bootstrap
	 * @param train_data - Orginal data
	 * @return  - Bootstrapped data
	 */
	private static TrainData getBootstrappedData( TrainData train_data )
	{
		TrainData baggedData = new TrainData();
		for ( int j = 0; j < train_data.attribute_list.size(); j++ )
		{
			Attribute attr = train_data.attribute_list.get( j );
			baggedData.addAttribute( attr.index, attr.name, attr.value_count );
			baggedData.addAttributeValues( j, attr.values );
		}
		baggedData.addClassAttributeValues( train_data.class_data );
		int size = train_data.class_data.attribute_data.size() - 1;
		Random rand = new Random();
		for ( int i = 0; i <= size; i++ )
		{
			int rand_val = rand.nextInt( size + 1 );
			for ( int j = 0; j < train_data.attribute_list.size(); j++ )
			{
				int val = train_data.attribute_list.get( j ).attribute_data.get( rand_val );
				baggedData.copyTrainData( j, val );
			}

			baggedData.copyClassTrainData( train_data.class_data.attribute_data.get( rand_val ) );
		}

		return baggedData;
	}
}
