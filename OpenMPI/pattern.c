#include <mpi.h>
#include <stdio.h>
#include <string.h>

//naive pattern match
int pattern_match(const char* sequence, const char* pattern, int seq_length, int pat_length);
int process_size; //number of processes
int process_id;   //process rank(id)


int main()
{
	const char* sequence = "GAATTGAATTCAGGATCGAGTTACAGTTAAATTCAGTTACGGATCGAAGTTA\n\
                            AGTTAAGTTAGAATATTCAGTGGATCGATACAGTTAAATTCAGTTACACAGT\n\
                            TGGATCGAAAGTTAAGTTAGAATATTCAGTTAGGAATTCAGGGATCGATTAC\n\
                            AGTTAAATTCAGTTTTAAGTTAATCAGTTAC";
	const char* pattern_string = "GGATCGA";

	int sequence_len = (int)strlen(sequence); 			  //sequence length
	int pattern_string_len = (int)strlen(pattern_string); //pattern length

	int total_matches = 0; //global count
	int local_matches = 0; //local count

	MPI_Init(NULL, NULL); // Initialize MPI
	MPI_Comm_size(MPI_COMM_WORLD, &process_size); //Get the number of processes
	MPI_Comm_rank(MPI_COMM_WORLD, &process_id);	  //Get the process rank(id)

	MPI_Status status;

	local_matches = pattern_match(sequence, pattern_string, sequence_len, pattern_string_len);

	if(process_id != 0) {
		MPI_Send(&local_matches, 1, MPI_INT, 0, 0, MPI_COMM_WORLD );
	}
	else {
		printf("Process rank: %d...total matches found: %d\n", process_id,local_matches);
		total_matches+=local_matches; //add local match count to global match count

		//Gather number of local matches
		for(int process_source = 1; process_source < process_size; process_source++) {
			MPI_Recv(&local_matches, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			printf("Receiving from process rank: %d...total matches found: %d\n", status.MPI_SOURCE, local_matches);
			total_matches+=local_matches; //add local match count to global match count
		}

		printf("TOTAL NUMBER OF MATCHES: %d\n", total_matches);
	}

	// MPI_Reduce(&local_matches, &total_matches, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	// if(process_id == 0) {
	// 	printf("Total matches: %d\n", total_matches);
	// }

	MPI_Finalize(); //Shut down MPI
	return 0;

}

int pattern_match(const char* sequence, const char* pattern, int seq_length, int pat_length)
{
  int idx = process_id;     //process id
  int localCount = 0;       //local match count
  int i = 0;

  while(idx <= seq_length-pat_length){
    if(pattern[i] != sequence[idx+i] || i == pat_length){
      if(i == pat_length)
        localCount++; //if i == pat_length, add 1 to localCount
      i = -1; //reset i variable
      idx += process_size; //increment our thread index
    }
    i++;
  }
  return localCount;
}