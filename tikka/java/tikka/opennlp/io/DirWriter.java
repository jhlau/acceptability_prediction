///////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2010 Taesun Moon, The University of Texas at Austin
//
//  This library is free software; you can redistribute it and/or
//  modify it under the terms of the GNU Lesser General Public
//  License as published by the Free Software Foundation; either
//  version 3 of the License, or (at your option) any later version.
//
//  This library is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU Lesser General Public License for more details.
//
//  You should have received a copy of the GNU Lesser General Public
//  License along with this program; if not, write to the Free Software
//  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
///////////////////////////////////////////////////////////////////////////////
package tikka.opennlp.io;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.Vector;

/**
 * Class object for walking through and writing to annotated files that mirror
 * the structure of the input directory.
 * 
 * @author tsmoon
 */
public class DirWriter {

    /**
     * Writer object to return at each iteration
     */
    protected BufferedWriter outputWriter;
    /**
     * Name of current file being written to
     */
    protected String currentFileName;
    /**
     * Index of current file being written to
     */
    protected Integer currentFileIdx = 0;
    /**
     * Vector of files in input with full path
     */
    protected Vector<String> files;
    /**
     * Root of input files
     */
    protected String inRoot;
    /**
     * Root of output files
     */
    protected String outRoot;

    public DirWriter(String outRoot, DirReader dirReader) throws IOException {
        files = new Vector<String>();
        inRoot = dirReader.root;
        this.outRoot = outRoot;
        for (String file : dirReader.files) {
            file.replaceFirst(inRoot, outRoot);
            files.add(file);
        }
    }

    public DirWriter(String outRoot, String inRoot, DirReader dirReader) throws
            IOException {
        files = new Vector<String>();
        this.inRoot = inRoot;
        this.outRoot = outRoot;
        for (String file : dirReader.files) {
            file = file.replaceFirst(inRoot, outRoot);
            files.add(file);
        }
    }

    public BufferedWriter nextOutputBuffer() {
        try {
            if (currentFileIdx < files.size()) {
                currentFileName = files.elementAt(currentFileIdx);
                outputWriter = new BufferedWriter(new OutputStreamWriter(
                        new FileOutputStream(currentFileName)));
                currentFileIdx++;
                return outputWriter;
            } else {
                return null;
            }

        } catch (IOException e) {
            File f = new File(currentFileName);
            String parent = f.getParent();
            (new File(parent)).mkdir();
            return nextOutputBuffer();
        }
    }

    public void reset() {
        currentFileIdx = 0;
    }

    public String getRoot() {
        return outRoot;
    }
}
