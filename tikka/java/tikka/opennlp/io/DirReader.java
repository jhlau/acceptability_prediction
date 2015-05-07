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

import java.io.File;
import java.io.IOException;
import java.util.Vector;

/**
 * Class object for walking through directory with training files.
 * 
 * @author tsmoon
 */
public class DirReader {

    /**
     * File handle of current file being read from
     */
    protected File currentFile;
    /**
     * Index of current file being read from
     */
    protected Integer currentFileIdx = 0;
    /**
     * Vector of files in input with full path
     */
    protected Vector<String> files;
    /**
     * Reader for the specific format
     */
    protected DataReader dataReader;
    /**
     * Format of input data (CONLL2K for now)
     */
    protected DataFormatEnum.DataFormat dataFormat;
    /**
     * Root of input files
     */
    protected String root;

    public DirReader() {
    }

    public DirReader(String root) throws IOException {
        if (!(new File(root)).isDirectory()) {
            throw new IOException();
        }
        this.root = root;
        files = new Vector<String>();
        walk(root);
    }

    public DirReader(String root, DataFormatEnum.DataFormat dataFormat) throws
            IOException {
        if (!(new File(root)).isDirectory()) {
            throw new IOException();
        }

        files = new Vector<String>();
        this.dataFormat = dataFormat;
        walk(root);
    }

    void walk(String root) {

        String[] paths = (new File(root)).list();
        for (String pathname : paths) {
            String currentpath = root + File.separator + pathname;
            if ((new File(currentpath)).isFile()) {
                files.add(currentpath);
            } else if ((new File(currentpath)).isDirectory()) {
                walk(currentpath);
            }
        }
    }

    public DataReader nextDocumentReader() {
        try {
            if (currentFileIdx < files.size()) {
                currentFile = new File(files.elementAt(currentFileIdx));
                currentFileIdx++;

                switch (dataFormat) {
                    case CONLL2K:
                        dataReader = new Conll2kReader(currentFile);
                        break;
                }

                return dataReader;
            } else {
                return null;
            }

        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    public void reset() {
        currentFileIdx = 0;
    }

    public String getRoot() {
        return root;
    }
}
