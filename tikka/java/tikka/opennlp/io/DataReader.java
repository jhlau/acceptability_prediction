///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2007 Jason Baldridge, The University of Texas at Austin
// 
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// 
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public
// License along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
//////////////////////////////////////////////////////////////////////////////
package tikka.opennlp.io;

import java.io.BufferedReader;
import java.io.EOFException;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.zip.GZIPInputStream;

/**
 * Wrapper for reading in information from a file which has one word
 * per line with associated tags. Subclasses for this abstract class
 * handle specific formats for data stored in this manner.
 *
 * @author  Jason Baldridge
 * @version $Revision: 1.53 $, $Date: 2006/10/12 21:20:44 $
 */
public abstract class DataReader {

    BufferedReader inputReader;
    File inputFile;

    protected DataReader (File f) throws IOException {
	inputFile = f;
	restart();
    }

    public abstract String[] nextToken() throws IOException, EOFException;
    public abstract String[][] nextSequence() throws IOException, EOFException;
    public abstract String[] nextOutputSequence() throws IOException, EOFException;

    // Override this in case a format needs to be set to a particular
    // spot for the first read (see HashSlashReader).x
    protected void prepare () throws IOException {}

    public void restart() throws IOException {
	if (inputFile.getName().endsWith(".gz"))
	    inputReader = new BufferedReader(new InputStreamReader(
  	                    new GZIPInputStream(new FileInputStream(inputFile))));
	else
	    inputReader = new BufferedReader(new FileReader(inputFile));

	prepare();
    }

    public File setFile(File f) throws IOException {
        inputFile = f;
        restart();
        return inputFile;
    }
    
    public void close() throws IOException {
	inputReader.close();
    }

    public String getFilename () {
	return inputFile.toString();
    }
}
