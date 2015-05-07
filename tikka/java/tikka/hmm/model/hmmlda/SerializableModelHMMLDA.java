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
package tikka.hmm.model.hmmlda;

import tikka.hmm.apps.CommandLineOptions;
import tikka.hmm.model.base.SerializableModel;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * Object where model parameters are saved. Includes both constant parameters
 * and inferred parameters.
 * 
 * @author tsmoon
 */
public class SerializableModelHMMLDA extends SerializableModel {

    /**
     * Hyperparameter for topic-by-document prior.
     */
    protected double alpha;
    /**
     * Hyperparameter for word/stem-by-topic prior
     */
    protected double beta;
    /**
     * Number of topic types.
     */
    protected int topicK;
    /**
     * Array of topic indexes. Of length {@link #wordN}.
     */
    protected int[] topicVector;

    /**
     * Constructor to use when model is being saved.
     * 
     * @param hmmlda Model to be saved
     */
    public SerializableModelHMMLDA(HMMLDA m) {
        super(m);
        alpha = m.alpha;
        beta = m.beta;
        topicK = m.topicK;
        topicVector = m.topicVector;
    }

    /**
     * Constructor to use when model is being loaded
     */
    public SerializableModelHMMLDA() {
    }

    /**
     * Load a previously trained model.
     *
     * @param filename  Full path of model location.
     * @return  The model that has been loaded.
     * @throws IOException
     * @throws FileNotFoundException
     */
    @Override
    public HMMLDA loadModel(CommandLineOptions options, String filename)
          throws IOException,
          FileNotFoundException {
        ObjectInputStream modelIn =
              new ObjectInputStream(new GZIPInputStream(new FileInputStream(
              filename)));
        try {
            loadBuffer = (SerializableModel) modelIn.readObject();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }
        copy(loadBuffer);
        loadBuffer = null;
        modelIn.close();

        HMMLDA hmmlda = new HMMLDA(options);

        return copy(hmmlda);
    }

    /**
     * Save the trained model.
     *
     * @param filename  Full path of model location.
     * @throws IOException
     */
    @Override
    public void saveModel(String filename) throws IOException {
        ObjectOutputStream modelOut =
              new ObjectOutputStream(new GZIPOutputStream(
              new FileOutputStream(filename)));
        modelOut.writeObject(this);
        modelOut.close();
    }

    protected void copy(SerializableModelHMMLDA sm) {
        super.copy(sm);
        alpha = sm.alpha;
        beta = sm.beta;
        topicK = sm.topicK;
        topicVector = sm.topicVector;
    }

    protected HMMLDA copy(HMMLDA hmmlda) {
        super.copy(hmmlda);
        hmmlda.alpha = alpha;
        hmmlda.beta = beta;
        hmmlda.topicK = topicK;
        hmmlda.topicVector = topicVector;

        return hmmlda;
    }
}
