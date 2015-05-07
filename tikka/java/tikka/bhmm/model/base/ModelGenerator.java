///////////////////////////////////////////////////////////////////////////////
// To change this template, choose Tools | Templates
// and open the template in the editor.
///////////////////////////////////////////////////////////////////////////////
package tikka.bhmm.model.base;

import tikka.bhmm.apps.CommandLineOptions;
import tikka.bhmm.models.*;

/**
 *
 * @author tsmoon
 */
public class ModelGenerator {

    public static HMMBase generator(CommandLineOptions options) {
        String modelName = options.getExperimentModel();
        return generator(modelName, options);
    }

    public static HMMBase generator(String modelName, CommandLineOptions options) {
        HMMBase bhmm = null;
//        if (modelName.equals("m1")) {
//            bhmm = new CDHMMS(options);
        if (modelName.equals("m2")) {
            bhmm = new ParHMM(options);
//        } else if (modelName.equals("m3")) {
//            bhmm = new HMMP(options);
        } else if (modelName.equals("m4")) {
            bhmm = new ParLDAHMM(options);
//        } else if (modelName.equals("m5")) {
//            bhmm = new CDHMMS(options);
        } else if (modelName.equals("m6")) {
            bhmm = new ParCDHMMD(options);
        } else if (modelName.equals("m7")) {
            bhmm = new ParHMMTwoTier(options);
//        } else if (modelName.equals("m8")) {
//            bhmm = new ParAdaptorHMM(options);
//        } else if (modelName.equals("m9")) {
//            bhmm = new ParNewmanSegmenter(options);
        } else if (modelName.equals("m10")) {
            bhmm = new ParBayesianChunker(options);
        }
        return bhmm;
    }
}
