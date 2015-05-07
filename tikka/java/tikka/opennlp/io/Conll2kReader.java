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

import java.io.EOFException;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

/**
 * Wrapper for reading in information from a file which has one word
 * per line with associated tags. E.g., the format used in the
 * CoNLL-2000 chunking data:
 *
 *   Confidence NN B-NP
 *   in IN B-PP
 *   the DT B-NP
 *   pound NN I-NP
 *   is VBZ B-VP
 *   widely RB I-VP
 *   expected VBN I-VP
 *   to TO I-VP
 *   take VB I-VP
 *   another DT B-NP
 *   sharp JJ I-NP
 *   dive NN I-NP
 *
 * @author  Jason Baldridge
 * @version $Revision: 1.53 $, $Date: 2006/10/12 21:20:44 $
 */
public class Conll2kReader extends DataReader {

    public Conll2kReader(File f) throws IOException {
        super(f);
    }

    public String[] nextToken() throws IOException, EOFException {
        String line = inputReader.readLine();
        if (line == null) {
            throw new EOFException();
        }
        line = line.trim();
        if (line.length() == 0) {
            return nextToken();
        } else {
            return line.split("\t");
        }
    }

    public String[][] nextSequence() throws IOException, EOFException {

        ArrayList<String[]> sequence = new ArrayList<String[]>();

        String line = inputReader.readLine();
        if (line == null) {
            throw new EOFException();
        }
        line = line.trim();

        if (line.length() == 0) {
            return nextSequence();
        }

        while (line.length() != 0) {
            sequence.add(line.split("\t"));
            line = inputReader.readLine();
            if (line == null) {
                line = "";
            }
            line = line.trim();
        }

        String[][] sequenceFixed = new String[sequence.size()][];
        sequence.toArray(sequenceFixed);
        return sequenceFixed;
    }

    public String[] nextOutputSequence() throws IOException, EOFException {

        ArrayList<String> sequence = new ArrayList<String>();

        String line = inputReader.readLine();
        if (line == null) {
            throw new EOFException();
        }
        line = line.trim();

        if (line.length() == 0) {
            return nextOutputSequence();
        }

        while (line.length() != 0) {
            if (line.indexOf('\t') > 0) {
                sequence.add(line.substring(0, line.indexOf('\t')));
            } else {
                sequence.add(line);
            }

            line = inputReader.readLine();
            if (line == null) {
                line = "";
            }
            line = line.trim();
        }

        String[] sequenceFixed = new String[sequence.size()];
        sequence.toArray(sequenceFixed);
        return sequenceFixed;
    }
}
