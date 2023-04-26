function candidates=SearchCandidates(Patches, Mask, dictionary_lowres, im_best_manifolds, flags)

NN = flags.NN;
patchDimL = flags.patchDimL;
num_training = flags.num_training;
sigma = flags.sigma;

[~,~,num_channel,~]=size(im_best_manifolds);


[~, h, w] = size(Patches);
nSamples=size(dictionary_lowres,1);

for i=1:h
    for j=1:w
        candidates(i,j).idx=zeros(NN,1);
        candidates(i,j).patches=zeros(num_channel*patchDimL^2,1);
    end
end

for i=1:h
    idx=[];
    for j=1:w
        Dist=[];
        if flags.IsTemplateMatch

            patch     = Patches(:,i,j);
            patch     = reshape(patch,[patchDimL, patchDimL, num_channel]);

            for l=1:num_training
                temp_image = py.numpy.array(im_best_manifolds(:,:,:,l));
                temp_template = py.numpy.array(patch);

                temp_image = temp_image.astype(py.numpy.float32);
                temp_template = temp_template.astype(py.numpy.float32);

                temp_cost_map = py.cv2.matchTemplate(temp_image, temp_template, py.cv2.TM_SQDIFF);

                cost_map = double(temp_cost_map);

                query_y_coordinate = i/h;
                lower_pass_bound = query_y_coordinate - flags.half_band;
                upper_pass_bound = query_y_coordinate + flags.half_band;
                cost_map_height=size(cost_map,1);
                penalty = cost_map*0;

                if lower_pass_bound > 0
                    penalty(1:1+ceil(lower_pass_bound*cost_map_height),:) = Inf;
                end
                if upper_pass_bound < 1
                    penalty(ceil(upper_pass_bound*cost_map_height):end,:) = Inf;
                end

                cost_map = cost_map + penalty;

                if numel(cost_map)~=nSamples
                    error('numel(cost_map)~=nSamples...(%d vs %d)', ...
                        numel(cost_map), nSamples);
                end

                Dist(:,:,l) = cost_map;

            end

            Dist=Dist(:);
        end

        prob_dis = exp(-Dist/2/patchDimL^2/sigma^2);
        prob_dis = prob_dis/(sum(prob_dis, 'all'));
        sampled_idx=randsample(length(Dist), NN, true, prob_dis);
        candidate_dist=Dist(sampled_idx);
        [~, candidate_order] = sort(candidate_dist);
        idx=sampled_idx(candidate_order);

    end
    candidates(i,j).idx = idx;
end
end

